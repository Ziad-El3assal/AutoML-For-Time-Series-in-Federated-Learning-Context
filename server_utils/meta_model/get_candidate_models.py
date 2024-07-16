class GetCandidateModels():
    def get_models(self, meta_features):
        # Call meta model and get candidate_models
        models = ['Lasso', 'ElasticNetCV']
        return models