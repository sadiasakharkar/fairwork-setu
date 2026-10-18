import pandas as pd

def prepare_jobs(df, title_column, description_column, skills_column):
    required = [title_column, description_column, skills_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    jobs = df.copy()
    for column in required:
        jobs[column] = jobs[column].fillna("").astype(str).str.strip()

    jobs["job_text"] = (
        jobs[title_column] + " " +
        jobs[description_column] + " " +
        jobs[skills_column]
    ).str.strip()

    jobs = jobs[jobs["job_text"] != ""].copy()
    jobs = jobs.drop_duplicates(subset=["job_text"]).reset_index(drop=True)

    if jobs.empty:
        raise ValueError("No usable job descriptions were found.")

    return jobs

def prepare_user_profile(skills, experience="", preferred_role="",
                         preferred_location=""):
    return " ".join([
        str(preferred_role or ""),
        str(skills or ""),
        str(experience or ""),
        str(preferred_location or "")
    ]).strip()
