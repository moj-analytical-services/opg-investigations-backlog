# tests/test_logit_pipeline.py
from g7_assessment.data import generate_synthetic, basic_clean, engineer_intervals
from g7_assessment.modeling import fit_legal_review_logit_enet

def test_logit_enet_pipeline_fits():
    df = engineer_intervals(basic_clean(generate_synthetic(1200, seed=7)))
    pipe, metrics, Xte, yte = fit_legal_review_logit_enet(df)
    assert "auc" in metrics and metrics["auc"] > 0.5  # should beat random on synthetic
