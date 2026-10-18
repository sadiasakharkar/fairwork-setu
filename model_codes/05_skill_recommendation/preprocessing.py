import pandas as pd

def prepare_courses(df, title_column, description_column, skills_column):
    required = [title_column, description_column, skills_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    courses = df.copy()
    for column in required:
        courses[column] = (
            courses[column].fillna("").astype(str).str.strip()
        )

    courses["course_text"] = (
        courses[title_column] + " " +
        courses[description_column] + " " +
        courses[skills_column]
    ).str.strip()

    courses = courses[courses["course_text"] != ""]
    courses = courses.drop_duplicates(
        subset=["course_text"]
    ).reset_index(drop=True)

    if courses.empty:
        raise ValueError("No usable course descriptions found.")

    return courses

def prepare_skill_profile(current_skills, target_role="", interests=""):
    return " ".join([
        str(current_skills or ""),
        str(target_role or ""),
        str(interests or "")
    ]).strip()
