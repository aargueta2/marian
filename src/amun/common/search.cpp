#include <boost/timer/timer.hpp>
#include "common/search.h"
#include "common/sentences.h"
#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"
#include "common/history_dijkstra.h"

/*
#include "gpu/decoder/encoder_decoder.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/dl4mt/dl4mt.h"
#include "gpu/decoder/encoder_decoder_state.h"
#include "gpu/decoder/best_hyps.h"
*/

#define DEBUG 1
using namespace std;

namespace amunmt {

Search::Search(const God &god)
  : deviceInfo_(god.GetNextDevice()),
    scorers_(god.GetScorers(deviceInfo_)),
    filter_(god.GetFilter()),
    maxBeamSize_(god.Get<size_t>("beam-size")),
    normalizeScore_(god.Get<bool>("normalize")),
    bestHyps_(god.GetBestHyps(deviceInfo_)),
    batchSize_(god.Get<size_t>("mini-batch"))
{}


Search::~Search() {
#ifdef CUDA
  if (deviceInfo_.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo_.deviceId);
  }
#endif
}

void Search::CleanAfterTranslation()
{
  for (auto scorer : scorers_) {
    scorer->CleanUpAfterSentence();
  }
}

std::shared_ptr<Histories> Search::Translate(const Sentences& sentences) {
  boost::timer::cpu_timer timer;

  size_t vocabulary_size = scorers_[0]->GetVocabSize();
  size_t number_scorers = scorers_.size();
  cout << "Vocab size: " << vocabulary_size << " and number of scorers: " << number_scorers << endl;

  if (filter_) {
    FilterTargetVocab(sentences);
  }

  //Encode the input sentences  
  States states = Encode(sentences);
  //Create variable to store the next states on the generation
  States nextStates = NewStates();
  //I think this is used to store the size of remaining elemenrs on the beam? e.g. if EOS is found
  //in one element of the beam, then the beam size of the batch member decreases by 1
  std::vector<uint> beamSizes(sentences.size(), 1);
  // If we need the default max beam size use "maxBeamSize_"
  uint selected_beam_size = 20;//vocabulary_size;//200;

  //TODO: Figure out how much memory histories and prevHyps actually consume
  std::shared_ptr<Histories> histories(new Histories(sentences, normalizeScore_));
  Beam prevHyps = histories->GetFirstHyps();

  //Resize the cost vector to fit the modified beam size
  bestHyps_->resizeCosts((batchSize_ * selected_beam_size));

  for (size_t decoderStep = 0; decoderStep < 3 * sentences.GetMaxLength(); ++decoderStep) {
   cout << "Decoder Step " << decoderStep << endl;

   //TODO: Find ways to separate the *states[i].get<EDState>();
   if(decoderStep == 0){
    for (size_t i = 0; i < scorers_.size(); i++){

      #if DEBUG
      std::cout << "DECODING" << std::endl;
      #endif

      cout << "Size: " << states[0]->Debug(0) << " and " << nextStates[0]->Debug(0) << endl;
      cout << "-Size: " << states[0]->Debug(1) << " and " << nextStates[0]->Debug(1) << endl;
 
      //const EDState& edIn = states[i]->get<EDState>();
      scorers_[i]->Decode(*states[i], *nextStates[i], beamSizes);

      #if DEBUG
      std::cout << "DONE DECODING" << std::endl;
      #endif

    }

    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
        beamSize = selected_beam_size;
      }
    }

    #if DEBUG
    cerr << "beamSizes=" << Debug(beamSizes, 1) << endl;
    #endif


    bool hasSurvivors = CalcBeam(histories, beamSizes, prevHyps, states, nextStates,selected_beam_size);

    printf("Done Calc Beam \n\n");
    if (!hasSurvivors) {
      break;
    }
   }
   else{




for (size_t i = 0; i < scorers_.size(); i++){

      #if DEBUG
      std::cout << "DECODING" << std::endl;
      #endif

      cout << "Size: " << states[0]->Debug(0) << " and " << nextStates[0]->Debug(0) << endl;
      cout << "-Size: " << states[0]->Debug(1) << " and " << nextStates[0]->Debug(1) << endl;
 
      //for(int test = 0;test < selected_beam_size; test++){
        std::cout << "TEST " << std::endl;
        scorers_[i]->Decode(*states[i], *nextStates[i], beamSizes,0);//test);
      //}

      #if DEBUG
      std::cout << "DONE DECODING" << std::endl;
      #endif

    }



    bool hasSurvivors = CalcBeam(histories, beamSizes, prevHyps, states, nextStates,selected_beam_size);

    printf("Done Calc Beam \n\n");
    if (!hasSurvivors) {
      break;
    }



   }





  }

  CleanAfterTranslation();

  LOG(progress)->info("Search took {}", timer.format(3, "%ws"));
  return histories;
}

States Search::Encode(const Sentences& sentences) {
  States states;
  for (auto& scorer : scorers_) {
    scorer->Encode(sentences);
    auto state = scorer->NewState();
    scorer->BeginSentenceState(*state, sentences.size());
    states.emplace_back(state);
  }
  return states;
}

bool Search::CalcBeam(
    std::shared_ptr<Histories>& histories,
    std::vector<uint>& beamSizes,
    Beam& prevHyps,
    States& states,
    States& nextStates,
    uint custom_beam_size)
{
    size_t batchSize = beamSizes.size();
    Beams beams(batchSize);

    bestHyps_->CalcBeam(prevHyps, scorers_, filterIndices_, beams, beamSizes,custom_beam_size);

    #if DEBUG
    std::cout << "ADD BEAMS" << std::endl;
    #endif


    histories->Add(beams);


    #if DEBUG
    std::cout << "END ADDING BEAMS" << std::endl;
    #endif

    Beam survivors;
    for (size_t batchId = 0; batchId < batchSize; ++batchId) {
      int token_index = 0;
      for (auto& h : beams[batchId]) {

        std::cout << "Obtained word [" << token_index << "] " << h->GetWord() << std::endl;
        token_index++;
        if (h->GetWord() != EOS_ID) {
          survivors.push_back(h);
        } else {
          survivors.push_back(h);
          //--beamSizes[batchId];
        }
      }
    }

    std::cout << "-------------" << std::endl;

    if (survivors.size() == 0) {
      return false;
    }

    #if DEBUG
    std::cout << "ASSEMBLE STATE" << std::endl;
    #endif
    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    #if DEBUG
    std::cout << "DONE ASSEMBLING " << std::endl;
    #endif
    prevHyps.swap(survivors);
    #if DEBUG
    std::cout << "RETURN " << std::endl;
    #endif
    return true;
}



bool Search::CalcBeam(
    std::shared_ptr<Histories>& histories,
    std::vector<uint>& beamSizes,
    Beam& prevHyps,
    States& states,
    States& nextStates)
{
    size_t batchSize = beamSizes.size();
    Beams beams(batchSize);

#if DEBUG
    std::cout << "CALL THE CALLC BEAM CODE " << std::endl;
#endif
    bestHyps_->CalcBeam(prevHyps, scorers_, filterIndices_, beams, beamSizes);

#if DEBUG
    std::cout << "ADD BEAMS" << std::endl;
#endif
    histories->Add(beams);
#if DEBUG
    std::cout << "END ADDING BEAMS" << std::endl;
#endif

    Beam survivors;
    for (size_t batchId = 0; batchId < batchSize; ++batchId) {
      for (auto& h : beams[batchId]) {
        if (h->GetWord() != EOS_ID) {
          survivors.push_back(h);
        } else {
          //survivors.push_back(h);
          --beamSizes[batchId];
        }
      }
    }

    if (survivors.size() == 0) {
      return false;
    }


    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    prevHyps.swap(survivors);
    return true;
}



States Search::NewStates() const {
  States states;
  for (auto& scorer : scorers_) {
    states.emplace_back(scorer->NewState());
  }
  return states;
}

void Search::FilterTargetVocab(const Sentences& sentences) {
  size_t vocabSize = scorers_[0]->GetVocabSize();
  std::set<Word> srcWords;
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence& sentence = *sentences.at(i);
    for (const auto& srcWord : sentence.GetWords()) {
      srcWords.insert(srcWord);
    }
  }

  filterIndices_ = filter_->GetFilteredVocab(srcWords, vocabSize);
  for (auto& scorer : scorers_) {
    scorer->Filter(filterIndices_);
  }
}



}

