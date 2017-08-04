#include "history_dijkstra.h"

#include "sentences.h"

namespace amunmt {

History_Dijkstra::History_Dijkstra(size_t lineNo, bool normalizeScore, size_t maxLength)
  : normalize_(normalizeScore),
    lineNo_(lineNo),
   maxLength_(maxLength)
{
  Add({HypothesisPtr(new Hypothesis())});
}


Histories_Dijkstra::Histories_Dijkstra(const Sentences& sentences, bool normalizeScore)
 : coll_(sentences.size())
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i).get();
    History_Dijkstra *history = new History_Dijkstra(sentence.GetLineNum(), normalizeScore, 3 * sentence.size());
    coll_[i].reset(history);
  }
}


class LineNumOrderer
{
  public:
    bool operator()(const std::shared_ptr<History_Dijkstra>& a, const std::shared_ptr<History_Dijkstra>& b) const
    {
      return a->GetLineNum() < b->GetLineNum();
    }
};


void Histories_Dijkstra::SortByLineNum()
{
  std::sort(coll_.begin(), coll_.end(), LineNumOrderer());
}


void Histories_Dijkstra::Append(const Histories_Dijkstra &other)
{
  for (size_t i = 0; i < other.size(); ++i) {
    std::shared_ptr<History_Dijkstra> history = other.coll_[i];
    coll_.push_back(history);
  }
}

}

