from collections import Counter
from typing import cast
from server_utils.aggregators.base_aggregator import Aggregator
import json
import flwr as fl
class XGBoostWeightsAggregator(Aggregator):

    def __init__(self):
        self.global_model = None

    def aggregate(self, parameters, data_sizes=[]):
        global_model = self.global_model

        for _, fit_res in parameters:
            update = fit_res.parameters.tensors
            for bst in update:
                global_model = self.__aggregate(global_model, bst)

        self.global_model = global_model
        parameters =  (
            fl.common.Parameters(tensor_type="xgboost_weights", tensors=[cast(bytes, global_model)]))
        return parameters
    def __aggregate(self,bst_prev_org,bst_curr_org):
        if not bst_prev_org:
            return bst_curr_org

        # Get the tree numbers
        tree_num_prev, _ = self._get_tree_nums(bst_prev_org)
        _, paral_tree_num_curr = self._get_tree_nums(bst_curr_org)

        bst_prev = json.loads(bytearray(bst_prev_org))
        bst_curr = json.loads(bytearray(bst_curr_org))

        bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ] = str(tree_num_prev + paral_tree_num_curr)
        iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
            "iteration_indptr"
        ]
        bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
            iteration_indptr[-1] + paral_tree_num_curr
        )

        # Aggregate new trees
        trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
        for tree_count in range(paral_tree_num_curr):
            trees_curr[tree_count]["id"] = tree_num_prev + tree_count
            bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
                trees_curr[tree_count]
            )
            bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

        bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")
        return bst_prev_bytes


    def _get_tree_nums(self,xgb_model_org):
        xgb_model = json.loads(bytearray(xgb_model_org))
        # Get the number of trees
        tree_num = int(
            xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_trees"
            ]
        )
        # Get the number of parallel trees
        paral_tree_num = int(
            xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_parallel_tree"
            ]
        )
        return tree_num, paral_tree_num
