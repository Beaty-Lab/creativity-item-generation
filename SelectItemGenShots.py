import itertools
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from statistics import mean


def SelectItemGenShots(
    itemPool: pd.DataFrame,
    shotSelectionMetric: str,
    itemGenNumShots: int,
    round: int,
    shotSelectionSort: str,
    shotSelectionAggregate: str,
    # if not using constraint satisfaction, defaults to a greedy method that just selects items based on originality scores
    useConstraintSatisfaction: bool = False,
):
    itemPool = pd.read_json(f"{itemPool}_round_{round}.json", orient="records")
    if useConstraintSatisfaction:
        return ConstraintSatisfaction(
            itemPool,
            shotSelectionMetric,
            itemGenNumShots,
            round,
            shotSelectionSort,
            shotSelectionAggregate,
        )
    else:
        # use greedy item selection
        if shotSelectionAggregate == "mean":
            meanItemScores = itemPool.groupby(f"creative_scenario_round_{round}").mean(
                numeric_only=True
            )
        elif shotSelectionAggregate == "variance":
            meanItemScores = itemPool.groupby(f"creative_scenario_round_{round}").var(
                numeric_only=True
            )

        if shotSelectionSort == "max":
            meanItemScores.sort_values(
                by=f"{shotSelectionMetric}_round_{round}", ascending=False, inplace=True
            )
        elif shotSelectionSort == "min":
            meanItemScores.sort_values(
                by=f"{shotSelectionMetric}_round_{round}", ascending=True, inplace=True
            )

        item_list = meanItemScores.iloc[:itemGenNumShots].index.to_list()
        return item_list


# debug constrained optimization of item gen selection using results from prior greedy runs
# will always use the greedy selection from the prior round to seed the select paramters
def ConstraintSatisfaction(
    itemPool: pd.DataFrame,
    shotSelectionMetric: str,
    itemGenNumShots: int,
    round: int,
    shotSelectionSort: str,
    shotSelectionAggregate: str,
    delta: float = 0.1,
):
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )  # TODO: make an arg for config

    itemPoolDf = pd.read_json(f"{itemPool}_round_{round}.json", orient="records")
    if round != 0:
        priorItemPool = pd.read_json(
            f"{itemPool}_round_{round-1}.json", orient="records"
        )
        
        if shotSelectionAggregate == "mean":
            priorMeanItemScores = priorItemPool.groupby(
                f"creative_scenario_round_{round-1}"
            ).mean(numeric_only=True)
        elif shotSelectionAggregate == "var":
            priorMeanItemScores = priorItemPool.groupby(
                f"creative_scenario_round_{round-1}"
            ).var(numeric_only=True)
        
        priorMeanItemScores.sort_values(
            by=f"{shotSelectionMetric}_round_{round-1}", ascending=False, inplace=True
        )
        prior_items = priorMeanItemScores.iloc[:itemGenNumShots]
        prior_originality = prior_items[
            f"{shotSelectionMetric}_round_{round-1}"
        ].values.mean()
        prior_item_list = priorMeanItemScores.iloc[:itemGenNumShots].index.to_list()

        prior_embeddings = embedding_model.encode(
            prior_item_list,
            convert_to_tensor=True,
        )
        prior_sims = util.cos_sim(prior_embeddings, prior_embeddings)
        # remove the diagonal and keep only the lower triangle since the matrix is symmetric
        prior_sims = prior_sims.tril().flatten()
        prior_sim_score = prior_sims[prior_sims != 0].cpu().numpy().mean()

    else:
        # default to minimum values for the first iteration
        prior_sim_score = 1.0
        prior_originality = -2.0
    
    if shotSelectionAggregate == "mean":
        MeanItemScores = itemPoolDf.groupby(f"creative_scenario_round_{round}").mean(
            numeric_only=True
        )
    elif shotSelectionAggregate == "var":
        MeanItemScores = itemPoolDf.groupby(f"creative_scenario_round_{round}").var(
            numeric_only=True
        )
    
    all_item_combs = list(
        itertools.combinations(MeanItemScores.index.to_list(), itemGenNumShots)
    )
    for item_comb in tqdm(
        all_item_combs, total=len(all_item_combs)
    ):  # dangerous, could run out of memory!
        item_embeddings = embedding_model.encode(item_comb, convert_to_tensor=True)
        sims = util.cos_sim(item_embeddings, item_embeddings)
        # remove the diagonal and keep only the lower triangle since the matrix is symmetric
        sims = sims.tril().flatten()

        sim_score = sims[sims != 0].cpu().numpy().mean()
        originality_scores = []
        for item in item_comb:
            originality_scores.append(
                MeanItemScores.loc[item][f"{shotSelectionMetric}_round_{round}"]
            )

        originality = mean(
            originality_scores
        )  # TODO: need to support other metrics besides mean

        print(f"# Sim: {sim_score} Prior Sim: {prior_sim_score} # Score: {originality} Prior Score: {prior_originality} #")

        if sim_score <= prior_sim_score and (
            originality >= prior_originality
            or ((prior_originality - originality) <= delta)
        ):
            return item_comb

    # unlikely, but there may not be an item set that satisfies the constraints
    # default to greedy approach
    print("Failed to locate item set satisfying constraints, defaulting to greedy selection.")
    return SelectItemGenShots(
        itemPool,
        shotSelectionMetric,
        itemGenNumShots,
        round,
        shotSelectionSort,
        shotSelectionAggregate,
        useConstraintSatisfaction=False,
    )
