#include "encoder.h"

using namespace std;

namespace amunmt {
namespace FPGA {

Encoder::Encoder(const cl_context &context, const Weights& model)
: embeddings_(model.encEmbeddings_)
, forwardRnn_(context, model.encForwardGRU_)
, backwardRnn_(context, model.encBackwardGRU_)
, Context(context)
, context_(context)
{

}

size_t GetMaxLength(const Sentences& source, size_t tab) {
  size_t maxLength = source.at(0)->GetWords(tab).size();
  for (size_t i = 0; i < source.size(); ++i) {
    const Sentence &sentence = *source.at(i);
    maxLength = std::max(maxLength, sentence.GetWords(tab).size());
  }
  return maxLength;
}


std::vector<std::vector<size_t>> GetBatchInput(const Sentences& source, size_t tab, size_t maxLen) {
  std::vector<std::vector<size_t>> matrix(maxLen, std::vector<size_t>(source.size(), 0));

  for (size_t j = 0; j < source.size(); ++j) {
    for (size_t i = 0; i < source.at(j)->GetWords(tab).size(); ++i) {
        matrix[i][j] = source.at(j)->GetWords(tab)[i];
    }
  }

  return matrix;
}

void Encoder::GetContext(const Sentences& source, size_t tab, mblas::Matrix& Context)
{
  size_t maxSentenceLength = GetMaxLength(source, tab);

  Context.Resize(maxSentenceLength * source.size(),
                 forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength());

  auto input = GetBatchInput(source, tab, maxSentenceLength);

  for (size_t i = 0; i < input.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back(context_);
    }
    embeddings_.Lookup(embeddedWords_[i], input[i]);
  }

  forwardRnn_.GetContext(embeddedWords_.cbegin(),
                         embeddedWords_.cbegin() + maxSentenceLength,
                         Context, source.size(), false);

  backwardRnn_.GetContext(embeddedWords_.crend() - maxSentenceLength,
                          embeddedWords_.crend() ,
                          Context, source.size(), true);

}

}
}