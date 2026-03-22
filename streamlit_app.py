from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai_job_dataset_merged_cleaned.csv"
CLS_PATH = BASE_DIR / "results" / "classification_results.csv"


def parse_skills(value: str) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Thiếu ai_job_dataset_merged_cleaned.csv. Hãy chạy Process.py trước.")

    df = pd.read_csv(DATA_PATH)
    for col in [
        "salary_usd",
        "salary_local",
        "years_experience",
        "remote_ratio",
        "skills_count",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["salary_usd"] = df["salary_usd"].fillna(df["salary_usd"].median())
    df["years_experience"] = df["years_experience"].fillna(df["years_experience"].median())
    df["remote_ratio"] = df["remote_ratio"].fillna(50)
    df["required_skills"] = df["required_skills"].fillna("")
    df["job_title"] = df["job_title"].fillna("Unknown")
    df["industry"] = df["industry"].fillna("Unknown")
    df["experience_level"] = df["experience_level"].fillna("Unknown")
    df["company_location"] = df["company_location"].fillna("Unknown")

    return df


@st.cache_data(show_spinner=False)
def get_high_salary_threshold(df: pd.DataFrame) -> float:
    if CLS_PATH.exists():
        cls_df = pd.read_csv(CLS_PATH)
        if "salary_threshold" in cls_df.columns and not cls_df.empty:
            return float(cls_df.loc[0, "salary_threshold"])
    return float(df["salary_usd"].quantile(0.75))


@st.cache_data(show_spinner=False)
def get_top_skills(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    all_skills: list[str] = []
    for item in df["required_skills"]:
        all_skills.extend(parse_skills(item))

    if not all_skills:
        return pd.DataFrame(columns=["skill", "count", "support"])

    vc = pd.Series(all_skills).value_counts().head(top_n)
    out = vc.rename_axis("skill").reset_index(name="count")
    out["support"] = out["count"] / max(len(df), 1)
    return out


@st.cache_data(show_spinner=False)
def get_skill_salary_uplift(df: pd.DataFrame, min_count: int = 40) -> pd.DataFrame:
    rows = []
    for _, row in df[["required_skills", "salary_usd"]].iterrows():
        skills = parse_skills(row["required_skills"])
        salary = float(row["salary_usd"])
        for skill in skills:
            rows.append((skill, salary))

    if not rows:
        return pd.DataFrame(columns=["skill", "count", "median_salary", "uplift"])

    temp = pd.DataFrame(rows, columns=["skill", "salary_usd"])
    base_salary = float(df["salary_usd"].median())
    agg = (
        temp.groupby("skill")["salary_usd"]
        .agg(count="count", median_salary="median")
        .reset_index()
    )
    agg = agg[agg["count"] >= min_count].copy()
    agg["uplift"] = agg["median_salary"] - base_salary
    agg = agg.sort_values("uplift", ascending=False)
    return agg


def infer_level(years: float) -> str:
    if years < 2:
        return "EN"
    if years < 5:
        return "MI"
    if years < 8:
        return "SE"
    return "EX"


def recommend_jobs(
    df: pd.DataFrame,
    years_exp: float,
    target_salary: float,
    remote_pref: int,
    selected_skills: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    inferred = infer_level(years_exp)
    cand = df.copy()

    cand = cand[cand["experience_level"].isin([inferred, "Unknown"]) | (cand["years_experience"] <= years_exp + 2)]
    cand = cand[(cand["salary_usd"] >= target_salary * 0.8)]
    cand = cand[(cand["remote_ratio"] - remote_pref).abs() <= 50]

    if selected_skills:
        selected_set = {x.lower() for x in selected_skills}

        def skill_match_count(text: str) -> int:
            current = {s.lower() for s in parse_skills(text)}
            return len(current.intersection(selected_set))

        cand["skill_match"] = cand["required_skills"].apply(skill_match_count)
        cand = cand[cand["skill_match"] > 0]
    else:
        cand["skill_match"] = 0

    cand = cand.sort_values(["skill_match", "salary_usd"], ascending=[False, False])
    cols = [
        "job_title",
        "industry",
        "experience_level",
        "company_location",
        "remote_ratio",
        "salary_usd",
        "required_skills",
        "skill_match",
    ]
    cols = [c for c in cols if c in cand.columns]
    return cand[cols].head(top_k)


def build_bot_response(
    df: pd.DataFrame,
    years_exp: float,
    target_salary: float,
    remote_pref: int,
    selected_skills: list[str],
    high_salary_threshold: float,
    uplift_df: pd.DataFrame,
) -> str:
    inferred = infer_level(years_exp)
    base = df[df["experience_level"] == inferred]["salary_usd"]
    base_median = float(base.median()) if not base.empty else float(df["salary_usd"].median())

    skill_bonus = 0.0
    if selected_skills and not uplift_df.empty:
        match = uplift_df[uplift_df["skill"].isin(selected_skills)]
        if not match.empty:
            skill_bonus = float(match["uplift"].clip(lower=0).mean())

    expected_salary = base_median + skill_bonus

    salary_score = 0
    if expected_salary >= target_salary:
        salary_score = 40
    elif expected_salary >= target_salary * 0.9:
        salary_score = 30
    elif expected_salary >= target_salary * 0.8:
        salary_score = 20
    else:
        salary_score = 10

    remote_slice = df[(df["experience_level"] == inferred) & ((df["remote_ratio"] - remote_pref).abs() <= 25)]
    remote_score = 30 if len(remote_slice) >= 100 else 20 if len(remote_slice) >= 30 else 10

    skill_score = min(len(selected_skills) * 6, 30)
    total_score = int(salary_score + remote_score + skill_score)

    tier = "Can xem xet"
    if total_score >= 75:
        tier = "Rat phu hop"
    elif total_score >= 55:
        tier = "Phu hop"

    high_salary_prob = "Cao" if expected_salary >= high_salary_threshold else "Trung binh"

    lines = []
    lines.append("### Bot tư vấn việc làm")
    lines.append(f"- Cấp độ kinh nghiệm ước tính: **{inferred}**")
    lines.append(f"- Mức lương kỳ vọng theo dữ liệu: **{expected_salary:,.0f} USD**")
    lines.append(f"- Ngưỡng high-salary tham chiếu: **{high_salary_threshold:,.0f} USD**")
    lines.append(f"- Đánh giá tổng thể: **{tier}** (điểm {total_score}/100)")
    lines.append(f"- Khả năng vào nhóm high-salary: **{high_salary_prob}**")

    lines.append("")
    lines.append("**Gợi ý hành động:**")
    if expected_salary < target_salary:
        lines.append("- Tăng khả năng đạt mức lương mục tiêu bằng cách bổ sung kỹ năng có support cao (Python, SQL, TensorFlow).")
    if remote_pref == 100:
        lines.append("- Ưu tiên tìm việc remote full-time để tăng độ phù hợp với ưu tiên cá nhân.")
    if len(selected_skills) < 3:
        lines.append("- Nên bổ sung thêm 2-3 kỹ năng cốt lõi để tăng tỉ lệ match job.")
    lines.append("- Tập trung vào job title và industry có salary median cao trong bảng Job Explorer.")

    return "\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="Trợ lý nghề nghiệp AI", page_icon="💼", layout="wide")

    st.title("Trung tâm kiến thức việc làm AI + Bot tư vấn nghề nghiệp")
    st.caption("Web đơn giản để tìm hiểu dữ liệu việc làm AI và nhận tư vấn job phù hợp.")

    df = load_data()
    threshold = get_high_salary_threshold(df)
    top_skills = get_top_skills(df, top_n=20)
    uplift_df = get_skill_salary_uplift(df)

    tab1, tab2, tab3 = st.tabs(["Kiến thức dữ liệu", "Khám phá việc làm", "Bot tư vấn"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số lượng job", f"{len(df):,}")
        c2.metric("Lương trung vị", f"{df['salary_usd'].median():,.0f} USD")
        c3.metric("Ngưỡng high-salary", f"{threshold:,.0f} USD")
        c4.metric("Remote trung vị", f"{df['remote_ratio'].median():.0f}%")

        st.subheader("Lương trung vị theo cấp độ kinh nghiệm")
        exp_salary = (
            df.groupby("experience_level")["salary_usd"].median().sort_values(ascending=False)
        )
        st.bar_chart(exp_salary)

        st.subheader("Top kỹ năng xuất hiện")
        if top_skills.empty:
            st.info("Không tìm thấy thông tin kỹ năng trong dữ liệu.")
        else:
            st.dataframe(top_skills, width="stretch")
            st.bar_chart(top_skills.set_index("skill")["support"])

    with tab2:
        st.subheader("Lọc và tìm job theo nhu cầu")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            exp_choices = sorted([x for x in df["experience_level"].dropna().unique().tolist() if str(x) != "nan"])
            exp_filter = st.multiselect("Cấp độ kinh nghiệm", exp_choices, default=exp_choices)
            min_salary = st.slider(
                "Mức lương tối thiểu (USD)",
                int(df["salary_usd"].min()),
                int(df["salary_usd"].max()),
                int(df["salary_usd"].quantile(0.5)),
                step=1000,
            )

        with col_b:
            industries = sorted(df["industry"].dropna().unique().tolist())
            selected_ind = st.multiselect("Ngành", industries, default=[])
            remote_pick = st.select_slider(
                "Mức độ remote mong muốn",
                options=[0, 50, 100],
                value=50,
                key="explorer_remote_pref",
            )

        with col_c:
            keyword = st.text_input("Từ khóa (job title)", value="")
            skill_options = top_skills["skill"].tolist() if not top_skills.empty else []
            explorer_skills = st.multiselect("Kỹ năng mong muốn", skill_options, default=[])

        filtered = df.copy()
        if exp_filter:
            filtered = filtered[filtered["experience_level"].isin(exp_filter)]
        filtered = filtered[filtered["salary_usd"] >= min_salary]
        filtered = filtered[(filtered["remote_ratio"] - remote_pick).abs() <= 50]
        if selected_ind:
            filtered = filtered[filtered["industry"].isin(selected_ind)]
        if keyword.strip():
            filtered = filtered[filtered["job_title"].str.contains(keyword, case=False, na=False)]

        if explorer_skills:
            selected_set = {x.lower() for x in explorer_skills}

            def has_any_skill(text: str) -> bool:
                return len({s.lower() for s in parse_skills(text)}.intersection(selected_set)) > 0

            filtered = filtered[filtered["required_skills"].apply(has_any_skill)]

        st.write(f"Số job phù hợp: **{len(filtered):,}**")
        display_cols = [
            "job_title",
            "industry",
            "experience_level",
            "company_location",
            "remote_ratio",
            "salary_usd",
            "required_skills",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(200), width="stretch")

    with tab3:
        st.subheader("Bot gợi ý việc làm phù hợp")
        left, right = st.columns(2)
        with left:
            years_exp = st.slider("Số năm kinh nghiệm", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
            target_salary = st.number_input(
                "Mức lương mục tiêu (USD)",
                min_value=20000.0,
                max_value=400000.0,
                value=120000.0,
                step=5000.0,
            )

        with right:
            remote_pref = st.select_slider(
                "Mức độ remote mong muốn",
                options=[0, 50, 100],
                value=50,
                key="advisor_remote_pref",
            )
            bot_skills = st.multiselect(
                "Kỹ năng bạn đang có",
                options=top_skills["skill"].tolist() if not top_skills.empty else [],
                default=[],
            )

        if st.button("Tư vấn ngay"):
            response = build_bot_response(
                df=df,
                years_exp=years_exp,
                target_salary=target_salary,
                remote_pref=remote_pref,
                selected_skills=bot_skills,
                high_salary_threshold=threshold,
                uplift_df=uplift_df,
            )

            st.markdown(response)

            st.markdown("**Job đề xuất:**")
            rec_df = recommend_jobs(
                df=df,
                years_exp=years_exp,
                target_salary=target_salary,
                remote_pref=remote_pref,
                selected_skills=bot_skills,
                top_k=10,
            )

            if rec_df.empty:
                st.warning("Không tìm thấy job phù hợp với bộ lọc hiện tại. Thử giảm target salary hoặc mở rộng kỹ năng.")
            else:
                st.dataframe(rec_df, width="stretch")


if __name__ == "__main__":
    main()
