from g7_assessment.data import generate_synthetic, basic_clean, engineer_intervals, daily_backlog_series, aggregate_staffing
from g7_assessment.modeling import fit_backlog_glm, fit_legal_review_classifier

def test_glm_and_classifier():
    df = engineer_intervals(basic_clean(generate_synthetic(800, seed=2)))
    daily = daily_backlog_series(df)
    staff = aggregate_staffing(df)
    glm, design = fit_backlog_glm(daily, staff)
    assert hasattr(glm, "predict")
    clf, _ = fit_legal_review_classifier(df)
    assert hasattr(clf, "predict_proba")
