from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_DIR = Path("datasets")
CASE_PATH = DATA_DIR / "case_intelligence_baseline.csv"
AI_SUMMARY_PATH = DATA_DIR / "aggregate_postmortem_summary.txt"


st.set_page_config(
    page_title="Fraud Case Intelligence Console",
    page_icon=":mag:",
    layout="wide",
)


@st.cache_data
def load_case_data() -> pd.DataFrame:
    df = pd.read_csv(CASE_PATH)
    return enrich_case_data(df)


def enrich_case_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "risk_band" not in df.columns:
        df["risk_band"] = pd.cut(
            df["risk_score"],
            bins=[0.0, 0.30, 0.70, 1.00],
            labels=["low", "medium", "high"],
            include_lowest=True,
        ).astype(str)

    if "is_fp" not in df.columns and {"predicted_label", "y_true"}.issubset(df.columns):
        df["is_fp"] = (
            (df["predicted_label"] == 1) & (df["y_true"] == 0)
        ).astype(int)

    if "is_tp" not in df.columns and {"predicted_label", "y_true"}.issubset(df.columns):
        df["is_tp"] = (
            (df["predicted_label"] == 1) & (df["y_true"] == 1)
        ).astype(int)

    if "borderline_flag" not in df.columns:
        df["borderline_flag"] = df["risk_score"].between(0.40, 0.60).astype(int)

    numeric_cols = [
        "per_day_transactions",
        "per_week_unique_ips",
        "per_week_payment_method_change",
        "num_unique_delivery_addresses",
        "email_address_age_days",
        "per_day_devices_per_user",
    ]
    thresholds = {
        "high_transaction_velocity": (
            "per_day_transactions",
            df["per_day_transactions"].quantile(0.95)
            if "per_day_transactions" in df.columns
            else None,
        ),
        "many_unique_ips": (
            "per_week_unique_ips",
            df["per_week_unique_ips"].quantile(0.95)
            if "per_week_unique_ips" in df.columns
            else None,
        ),
        "payment_method_changes": (
            "per_week_payment_method_change",
            df["per_week_payment_method_change"].quantile(0.95)
            if "per_week_payment_method_change" in df.columns
            else None,
        ),
        "many_delivery_addresses": (
            "num_unique_delivery_addresses",
            df["num_unique_delivery_addresses"].quantile(0.95)
            if "num_unique_delivery_addresses" in df.columns
            else None,
        ),
        "device_volatility": (
            "per_day_devices_per_user",
            df["per_day_devices_per_user"].quantile(0.95)
            if "per_day_devices_per_user" in df.columns
            else None,
        ),
    }

    if "new_email_risk" not in df.columns and "email_address_age_days" in df.columns:
        low_email_age = df["email_address_age_days"].quantile(0.10)
        df["new_email_risk"] = (df["email_address_age_days"] <= low_email_age).astype(int)

    for flag_name, (source_col, threshold) in thresholds.items():
        if flag_name in df.columns:
            continue
        if source_col in numeric_cols and source_col in df.columns and threshold is not None:
            df[flag_name] = (df[source_col] >= threshold).astype(int)

    reason_cols = [
        "high_transaction_velocity",
        "many_unique_ips",
        "payment_method_changes",
        "many_delivery_addresses",
        "device_volatility",
        "new_email_risk",
    ]
    available_reason_cols = [col for col in reason_cols if col in df.columns]

    if "reason_code_combo" not in df.columns:
        def combine_reason_codes(row: pd.Series) -> str:
            active = [col for col in available_reason_cols if row[col] == 1]
            return ", ".join(active) if active else "no_reason_code"

        df["reason_code_combo"] = df.apply(combine_reason_codes, axis=1)

    if "root_cause_category" not in df.columns:
        def assign_root_cause(row: pd.Series) -> str:
            if row.get("high_transaction_velocity", 0) == 1:
                return "Velocity Abuse"
            if row.get("payment_method_changes", 0) == 1:
                return "Payment Manipulation"
            if row.get("many_unique_ips", 0) == 1 or row.get("device_volatility", 0) == 1:
                return "Device / IP Volatility"
            if row.get("many_delivery_addresses", 0) == 1:
                return "Address Abuse"
            if row.get("new_email_risk", 0) == 1:
                return "New User Risk"
            if row.get("borderline_flag", 0) == 1:
                return "Borderline / Ambiguous"
            return "Mixed Risk"

        df["root_cause_category"] = df.apply(assign_root_cause, axis=1)

    return df


def load_ai_summary() -> str | None:
    if AI_SUMMARY_PATH.exists():
        return AI_SUMMARY_PATH.read_text().strip()
    return None


def build_overview_summary(df: pd.DataFrame) -> str:
    total_cases = len(df)
    flagged = int(df["predicted_label"].sum())
    flagged_rate = flagged / total_cases if total_cases else 0

    top_root = (
        df["root_cause_category"].value_counts().index[0]
        if "root_cause_category" in df.columns and not df.empty
        else "Unknown"
    )

    fp_df = df[df["is_fp"] == 1] if "is_fp" in df.columns else pd.DataFrame()
    if not fp_df.empty and "root_cause_category" in fp_df.columns:
        top_friction = fp_df["root_cause_category"].value_counts().index[0]
    else:
        top_friction = "No clear friction segment yet"

    high_risk = df[df["risk_band"] == "high"] if "risk_band" in df.columns else pd.DataFrame()
    if not high_risk.empty and "latest_item_category" in high_risk.columns:
        top_item = high_risk["latest_item_category"].fillna("Missing").value_counts().index[0]
    else:
        top_item = "Unknown"

    return (
        f"The dashboard is currently tracking {total_cases:,} cases, with "
        f"{flagged:,} flagged users ({flagged_rate:.1%}). The dominant flagged pattern is "
        f"{top_root}, while the biggest potential friction cluster is {top_friction}. "
        f"Among high-risk cases, the most common item category is {top_item}. "
        f"Analysts should review whether that segment reflects true abuse or avoidable review burden."
    )


def build_case_summary(case: pd.Series) -> str:
    evidence = []
    for label in [
        ("high_transaction_velocity", "high transaction velocity"),
        ("many_unique_ips", "multiple unique IPs"),
        ("payment_method_changes", "frequent payment method changes"),
        ("many_delivery_addresses", "many delivery addresses"),
        ("device_volatility", "device volatility"),
        ("new_email_risk", "new-account risk"),
    ]:
        if case.get(label[0], 0) == 1:
            evidence.append(label[1])

    if not evidence:
        evidence_text = "mixed weaker signals rather than one dominant trigger"
    else:
        evidence_text = ", ".join(evidence[:3])

    return (
        f"Case {int(case['consumer_id'])} sits in the {case['risk_band']} risk band with a "
        f"risk score of {case['risk_score']:.3f}. The pattern is most consistent with "
        f"{case['root_cause_category']}, supported by {evidence_text}. "
        f"Recommended next step: review this case alongside similar users in the same segment "
        f"before deciding whether to escalate, step-up verify, or suppress."
    )


def metric_row(df: pd.DataFrame) -> None:
    total_cases = len(df)
    flagged_cases = int(df["predicted_label"].sum())
    fraud_rate = df["y_true"].mean() if "y_true" in df.columns else 0
    avg_score = df["risk_score"].mean()
    fp_rate = df["is_fp"].mean() if "is_fp" in df.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Cases", f"{total_cases:,}")
    col2.metric("Flagged Cases", f"{flagged_cases:,}")
    col3.metric("Observed Fraud Rate", f"{fraud_rate:.1%}")
    col4.metric("Average Risk Score", f"{avg_score:.3f}")
    col5.metric("False Positive Share", f"{fp_rate:.1%}")


def overview_page(df: pd.DataFrame) -> None:
    st.title("Fraud Case Intelligence Console")
    st.caption(
        "Surface what patterns are driving flags, where customer friction may be happening, "
        "and what ops teams should review next."
    )

    metric_row(df)
    st.markdown("---")

    insight_text = load_ai_summary() or build_overview_summary(df)
    st.subheader("AI Insight Summary")
    st.info(insight_text)

    left, right = st.columns(2)

    with left:
        st.subheader("Flagged Cases by Root Cause")
        flagged_root = (
            df[df["predicted_label"] == 1]["root_cause_category"]
            .value_counts()
            .reset_index()
        )
        flagged_root.columns = ["root_cause_category", "cases"]
        fig = px.bar(
            flagged_root,
            x="root_cause_category",
            y="cases",
            color="root_cause_category",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Cases")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Reason Codes")
        top_reasons = (
            df[df["predicted_label"] == 1]["reason_code_combo"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_reasons.columns = ["reason_code_combo", "cases"]
        st.dataframe(top_reasons, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Risk Band Distribution")
        risk_counts = df["risk_band"].value_counts().reindex(["low", "medium", "high"], fill_value=0)
        risk_df = risk_counts.reset_index()
        risk_df.columns = ["risk_band", "cases"]
        fig = px.bar(risk_df, x="risk_band", y="cases", color="risk_band")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Cases")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Flagged Item Categories")
        top_items = (
            df[df["predicted_label"] == 1]["latest_item_category"]
            .fillna("Missing")
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_items.columns = ["latest_item_category", "cases"]
        st.dataframe(top_items, use_container_width=True, hide_index=True)


def case_explorer_page(df: pd.DataFrame) -> None:
    st.title("Case Explorer")

    flagged_only = st.sidebar.checkbox("Flagged cases only", value=True)
    risk_band_options = st.sidebar.multiselect(
        "Risk bands",
        options=sorted(df["risk_band"].dropna().unique().tolist()),
        default=sorted(df["risk_band"].dropna().unique().tolist()),
    )
    root_options = st.sidebar.multiselect(
        "Root cause categories",
        options=sorted(df["root_cause_category"].dropna().unique().tolist()),
        default=sorted(df["root_cause_category"].dropna().unique().tolist()),
    )

    filtered = df.copy()
    if flagged_only:
        filtered = filtered[filtered["predicted_label"] == 1]
    filtered = filtered[filtered["risk_band"].isin(risk_band_options)]
    filtered = filtered[filtered["root_cause_category"].isin(root_options)]

    st.subheader("Case Table")
    display_cols = [
        "consumer_id",
        "risk_score",
        "risk_band",
        "predicted_label",
        "y_true",
        "root_cause_category",
        "reason_code_combo",
        "latest_item_category",
        "latest_delivery_address_region_label",
    ]
    available_cols = [col for col in display_cols if col in filtered.columns]
    st.dataframe(
        filtered[available_cols].sort_values("risk_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Selected Case Detail")
    if filtered.empty:
        st.warning("No cases match the current filters.")
        return

    case_ids = filtered["consumer_id"].astype(int).tolist()
    selected_id = st.selectbox("Select consumer_id", case_ids)
    selected_case = filtered[filtered["consumer_id"] == selected_id].iloc[0]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Risk Score", f"{selected_case['risk_score']:.3f}")
        st.metric("Risk Band", str(selected_case["risk_band"]).title())
        st.metric("Root Cause", selected_case["root_cause_category"])
        st.write("Reason Codes")
        st.code(selected_case["reason_code_combo"])

    with col2:
        raw_cols = [
            "per_day_transactions",
            "per_week_unique_ips",
            "per_week_payment_method_change",
            "num_unique_delivery_addresses",
            "email_address_age_days",
            "latest_item_category",
            "latest_item_product_title",
            "latest_delivery_address_region_label",
        ]
        raw_display = {
            col: selected_case[col]
            for col in raw_cols
            if col in selected_case.index
        }
        st.write("Raw Signals")
        st.json(raw_display)

    st.subheader("AI Case Summary")
    st.success(build_case_summary(selected_case))


def friction_page(df: pd.DataFrame) -> None:
    st.title("Friction Analysis")
    st.caption("Find where the system may be over-flagging or creating review burden.")

    left, right = st.columns(2)

    with left:
        st.subheader("False Positive Rate by Root Cause")
        root_summary = (
            df.groupby("root_cause_category")
            .agg(
                cases=("consumer_id", "count"),
                false_positives=("is_fp", "sum"),
                avg_risk_score=("risk_score", "mean"),
            )
            .reset_index()
        )
        root_summary["fp_rate"] = root_summary["false_positives"] / root_summary["cases"]
        root_summary = root_summary.sort_values("fp_rate", ascending=False)
        fig = px.bar(
            root_summary,
            x="fp_rate",
            y="root_cause_category",
            color="root_cause_category",
            orientation="h",
            hover_data=["cases", "false_positives", "avg_risk_score"],
        )
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("False Positives by Region")
        region_summary = (
            df.groupby("latest_delivery_address_region_label")
            .agg(
                cases=("consumer_id", "count"),
                false_positives=("is_fp", "sum"),
            )
            .reset_index()
        )
        region_summary["fp_rate"] = region_summary["false_positives"] / region_summary["cases"]
        region_summary = region_summary.sort_values("false_positives", ascending=False).head(15)
        st.dataframe(region_summary, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Borderline Cases")
        borderline = df[df["borderline_flag"] == 1].copy()
        borderline_display_cols = [
            "consumer_id",
            "risk_score",
            "predicted_label",
            "y_true",
            "root_cause_category",
            "reason_code_combo",
        ]
        borderline_cols = [col for col in borderline_display_cols if col in borderline.columns]
        st.dataframe(
            borderline[borderline_cols].sort_values("risk_score", ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Potential Review Burden Segments")
        burden = (
            df.groupby(["risk_band", "root_cause_category"])
            .agg(
                cases=("consumer_id", "count"),
                false_positives=("is_fp", "sum"),
            )
            .reset_index()
        )
        burden["fp_rate"] = burden["false_positives"] / burden["cases"]
        burden = burden.sort_values(["false_positives", "fp_rate"], ascending=False).head(15)
        st.dataframe(burden, use_container_width=True, hide_index=True)

    st.subheader("AI Postmortem")
    ai_summary = load_ai_summary()
    if ai_summary:
        st.info(ai_summary)
    else:
        fp_cases = int(df["is_fp"].sum())
        borderline_cases = int(df["borderline_flag"].sum())
        top_fp_root = (
            df[df["is_fp"] == 1]["root_cause_category"].value_counts().index[0]
            if fp_cases > 0
            else "No dominant false-positive cluster yet"
        )
        st.info(
            f"False positives currently account for {fp_cases:,} cases, with "
            f"{borderline_cases:,} users sitting near the decision threshold. "
            f"The most friction-prone cluster appears to be {top_fp_root}. "
            f"Recommended next step: review whether that segment should move from hard flagging "
            f"to step-up verification or analyst review."
        )


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Case Explorer", "Friction Analysis"],
    )

    if not CASE_PATH.exists():
        st.error(
            "Missing dataset: datasets/case_intelligence_baseline.csv. "
            "Run your notebook export first."
        )
        return

    df = load_case_data()

    if page == "Overview":
        overview_page(df)
    elif page == "Case Explorer":
        case_explorer_page(df)
    else:
        friction_page(df)


if __name__ == "__main__":
    main()
