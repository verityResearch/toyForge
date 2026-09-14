#include "toyforge/reward.h"

double tf_compute_reward(TfSubscores subscores, TfRubric rubric) {
  return rubric.parse * subscores.parse + rubric.schema * subscores.schema +
         rubric.method_known * subscores.method_known +
         rubric.precondition_met * subscores.precondition_met +
         rubric.transition_valid * subscores.transition_valid +
         rubric.sequence_optimal * subscores.sequence_optimal;
}
