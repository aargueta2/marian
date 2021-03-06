#pragma once

#include <map>
#include <numeric>
#include <boost/timer/timer.hpp>

#include "common/scorer.h"
#include "common/exception.h"
#include "common/god.h"
#include "common/utils.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/nth_element.h"

#include "gpu/decoder/encoder_decoder.h"

#define DEBUG 0
namespace amunmt {
namespace GPU {

class BestHyps : public BestHypsBase
{
  public:
    BestHyps(const BestHyps &copy) = delete;

    BestHyps(const God &god)
          : BestHypsBase(
              !god.Get<bool>("allow-unk"),
              god.Get<bool>("n-best"),
              god.Get<std::vector<std::string>>("softmax-filter").size(),
              god.Get<bool>("return-alignment") || god.Get<bool>("return-soft-alignment"),
              god.GetScorerWeights()),
            nthElement_(god.Get<size_t>("beam-size") , god.Get<size_t>("mini-batch")),
            keys(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
            Costs(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch"))
    {}

    void DisAllowUNK(mblas::Matrix& Prob) {
      SetColumn(Prob, UNK_ID, std::numeric_limits<float>::lowest());
    }

    void FindBests(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst) {
      nthElement_.getNBestList(beamSizes, Probs, outCosts, outKeys, isFirst);
    }


    void FindBests(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst,
                   uint custom_beam_size) {
      nthElement_.getNBestList(beamSizes, Probs, outCosts, outKeys, custom_beam_size,isFirst);
    }

    std::vector<SoftAlignmentPtr> GetAlignments(const std::vector<ScorerPtr>& scorers,
                                                size_t hypIndex) {
      std::vector<SoftAlignmentPtr> alignments;
      for (auto& scorer : scorers) {
        if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorer.get())) {
          const mblas::Matrix &attention = encdec->GetAttention();
          size_t attLength = attention.dim(1);

          SoftAlignment *softAlignment = new SoftAlignment(attLength);
          mblas::copy(
              attention.data() + hypIndex * attLength,
              attLength,
              thrust::raw_pointer_cast(softAlignment->data()),
              cudaMemcpyDeviceToHost
          );

          alignments.emplace_back(softAlignment);
        } else {
          amunmt_UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
        }
      }
      return alignments;
    }

    void CalcBeam(
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams,
        std::vector<uint>& beamSizes)
    {
      BEGIN_TIMER("CalcBeam");

      #if DEBUG
      std::cout << "CALCBEAM SECTION" << std::endl;
      #endif
      using namespace mblas;

      mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());

      HostVector<float> vCosts;
      for (auto& h : prevHyps) {
        vCosts.push_back(h->GetCost());
      }
      mblas::copy(vCosts.begin(), vCosts.end(), Costs.begin());

      const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

      BroadcastVecColumn(weights_.at(scorers[0]->GetName()) * _1 + _2, Probs, Costs);

      for (size_t i = 1; i < scorers.size(); ++i) {
        mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

        Element(_1 + weights_.at(scorers[i]->GetName()) * _2, Probs, currProbs);
      }

      if (forbidUNK_) {
        DisAllowUNK(Probs);
      }

      size_t beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

      std::vector<float> bestCosts;
      std::vector<unsigned> bestKeys;

      FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);

      std::vector<HostVector<float>> breakDowns;
      if (returnNBestList_) {
          breakDowns.push_back(bestCosts);
          for (size_t i = 1; i < scorers.size(); ++i) {
            std::vector<float> modelCosts(beamSizeSum);
            mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

            nthElement_.getValueByKey(modelCosts, currProbs);
            breakDowns.push_back(modelCosts);
          }
      }

      std::map<size_t, size_t> batchMap;
      size_t tmp = 0;
      for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
        for (size_t t = 0; t < beamSizes[batchID]; ++t) {
          batchMap[tmp++] = batchID;
        }
      }

      for (size_t i = 0; i < beamSizeSum; i++) {
        size_t wordIndex = bestKeys[i] % Probs.dim(1);
        if (isInputFiltered_) {
          wordIndex = filterIndices[wordIndex];
        }

        size_t hypIndex  = bestKeys[i] / Probs.dim(1);
        float cost = bestCosts[i];

        HypothesisPtr hyp;
        if (returnAttentionWeights_) {
          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
                                   GetAlignments(scorers, hypIndex)));
        } else {
          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
        }

        if(returnNBestList_) {
          hyp->GetCostBreakdown().resize(scorers.size());
          float sum = 0;
          for (size_t j = 0; j < scorers.size(); ++j) {
            if (j == 0)
              hyp->GetCostBreakdown()[0] = breakDowns[0][i];
            else {
              float cost = 0;
              if (j < scorers.size()) {
                  if (prevHyps[hypIndex]->GetCostBreakdown().size() < scorers.size())
                    const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(scorers.size(), 0.0f);
                  cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
              }
              sum += weights_.at(scorers[j]->GetName()) * cost;
              hyp->GetCostBreakdown()[j] = cost;
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights_.at(scorers[0]->GetName());
        }

        beams[batchMap[i]].push_back(hyp);
      }

      PAUSE_TIMER("CalcBeam");
    }


    void CalcBeam(
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams,
        std::vector<uint>& beamSizes,
        uint custom_beam_size)
    {
      BEGIN_TIMER("CalcBeam");
      #if DEBUG
      std::cout << "START CALC BEAM " << std::endl;
      #endif
      using namespace mblas;

      #if DEBUG
      std::cout << "GET PROBS " << std::endl;
      #endif
      mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());
      #if DEBUG
      std::cout << "GET VCOSTS " << std::endl;
      #endif

      HostVector<float> vCosts;
      for (auto& h : prevHyps) {
        #if DEBUG
        std::cout << "---COST " << std::endl;
        #endif
        vCosts.push_back(h->GetCost());
      }
      #if DEBUG
      std::cout << "COPY COSTS " << vCosts.size() << " - " << Costs.size() << std::endl;
      #endif


      mblas::copy(vCosts.begin(), vCosts.end(), Costs.begin());
#if DEBUG
std::cout << "IS FIRST " << std::endl;
#endif

      const bool isFirst = (vCosts[0] == 0.0f) ? true : false;
#if DEBUG
std::cout << "Get COL BY KEY " << std::endl;
#endif
      BroadcastVecColumn(weights_.at(scorers[0]->GetName()) * _1 + _2, Probs, Costs);
#if DEBUG
std::cout << "Get DONE COL BY KEY " << std::endl;
#endif
      for (size_t i = 1; i < scorers.size(); ++i) {
        mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

        Element(_1 + weights_.at(scorers[i]->GetName()) * _2, Probs, currProbs);
      }

      if (forbidUNK_) {
        DisAllowUNK(Probs);
      }

      size_t beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

      std::vector<float> bestCosts;
      std::vector<unsigned> bestKeys;

#if DEBUG
      std::cout << "-Get VALUE BY KEY " << std::endl;
#endif

      FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst, custom_beam_size);

#if DEBUG
      std::cout << "*Get VALUE BY KEY " << std::endl;
#endif

      std::vector<HostVector<float>> breakDowns;
      if (returnNBestList_) {
          breakDowns.push_back(bestCosts);
          for (size_t i = 1; i < scorers.size(); ++i) {
            std::vector<float> modelCosts(beamSizeSum);
            mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

            #if DEBUG
            std::cout << "Get VALUE BY KEY " << std::endl;
            #endif
            nthElement_.getValueByKey(modelCosts, currProbs);
            #if DEBUG
            std::cout << "DONE GETTING VALUE BY KEY " << std::endl;
            #endif
            breakDowns.push_back(modelCosts);
          }
      }

      std::map<size_t, size_t> batchMap;
      size_t tmp = 0;
      for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
        for (size_t t = 0; t < beamSizes[batchID]; ++t) {
          batchMap[tmp++] = batchID;
        }
      }

      for (size_t i = 0; i < beamSizeSum; i++) {
        size_t wordIndex = bestKeys[i] % Probs.dim(1);
        if (isInputFiltered_) {
          wordIndex = filterIndices[wordIndex];
        }

        size_t hypIndex  = bestKeys[i] / Probs.dim(1);
        float cost = bestCosts[i];

        std::cout << "Hypothesis index [" << i << "] " << hypIndex << " and cost" << cost << std::endl;
        HypothesisPtr hyp;
        if (returnAttentionWeights_) {
          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
                                   GetAlignments(scorers, hypIndex)));
        } else {
          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
        }

        if(returnNBestList_) {
          hyp->GetCostBreakdown().resize(scorers.size());
          float sum = 0;
          for (size_t j = 0; j < scorers.size(); ++j) {
            if (j == 0)
              hyp->GetCostBreakdown()[0] = breakDowns[0][i];
            else {
              float cost = 0;
              if (j < scorers.size()) {
                  if (prevHyps[hypIndex]->GetCostBreakdown().size() < scorers.size())
                    const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(scorers.size(), 0.0f);
                  cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
              }
              sum += weights_.at(scorers[j]->GetName()) * cost;
              hyp->GetCostBreakdown()[j] = cost;
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights_.at(scorers[0]->GetName());
        }

/*
        if(beams[batchMap[i]].size() > 0){
          beams[batchMap[i]][0] = hyp;
        }
        else{
          beams[batchMap[i]].push_back(hyp);
        }
*/
        beams[batchMap[i]].push_back(hyp);

        //printf("The size of %d is %d \n",batchMap[i],beams[batchMap[i]].size());
      }
      std::cout << "***********************" << std::endl;

      PAUSE_TIMER("CalcBeam");
    }

   void resizeCosts(uint size){
     Costs.resize(size);
   }

  private:
    NthElement nthElement_;
    DeviceVector<unsigned> keys;
    DeviceVector<float> Costs;
};

}
}

