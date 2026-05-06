from typing import Optional
import pandas as pd
import numpy as np
from typing import Literal, Callable


ApprovalScope = Literal["loo", "global"]
DFTransform = Callable[[pd.DataFrame], pd.DataFrame]


def get_user_preference_per_movie(data_matrix: pd.DataFrame) -> pd.DataFrame:
    user_matrix = data_matrix.groupby('userId')


def calc_metrics(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    df['global_approval'] = (
        np.tanh(df['preference_direction']/scale) > 0.75).astype(int) * (df['rating'] > 3.0).astype(int)
    return df


def sum_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df['group_sum'] = df.groupby('userId')['rating'].transform('sum')
    return df


def count_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df['group_count'] = df.groupby('userId')['rating'].transform('count')
    return df


def sum_squares_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df['group_sum_sq'] = df.groupby(
        'userId')['rating'].transform(lambda x: (x**2).sum())
    return df


def loo_mean(df: pd.DataFrame) -> pd.DataFrame:
    df['loo_mean'] = (df['group_sum'] - df['rating']) / (df['group_count'] - 1)
    return df


def global_mean(df: pd.DataFrame) -> pd.DataFrame:
    df['global_mean'] = df['group_sum'] / df['group_count']
    return df


def global_std(df: pd.DataFrame) -> pd.DataFrame:
    df['global_std'] = np.sqrt(
        df['group_sum_sq'] / df['group_count'] - df['global_mean']**2)
    return df


def loo_std(df: pd.DataFrame) -> pd.DataFrame:
    loo_sum = df['group_sum'] - df['rating']
    loo_sum_sq = df['group_sum_sq'] - (df['rating'] ** 2)
    loo_count = df['group_count'] - 1

    loo_var = (loo_sum_sq - (loo_sum ** 2 / loo_count)) / (loo_count - 1)
    df['loo_std'] = np.sqrt(loo_var.clip(lower=0))
    return df


def global_std(df: pd.DataFrame) -> pd.DataFrame:
    df['global_std'] = np.sqrt(
        df['group_sum_sq'] / df['group_count'] - df['global_mean']**2)
    return df


def global_preference_direction(df: pd.DataFrame) -> pd.DataFrame:
    df['global_preference_direction'] = (
        df['rating'] - df['global_mean']) / df['global_std']
    return df


def loo_preference_direction(df: pd.DataFrame) -> pd.DataFrame:
    df['loo_preference_direction'] = (
        df['rating'] - df['loo_mean']) / df['loo_std']
    return df


def make_approval(
    scope: ApprovalScope = "loo",
    scale: float = 1.0,
    rating_thresh: float = 3.0,
    tanh_thresh: float = 0.75,
    out_col: Optional[str] = None,
) -> DFTransform:
    prefix = "global_" if scope == "global" else "loo_"
    pref_col = f"{prefix}preference_direction"
    # "approval" or "global_approval"
    out_col = out_col or f"{prefix}approval"

    def _add(df: pd.DataFrame) -> pd.DataFrame:
        # Optional: fail fast with a nicer error than a KeyError
        missing = [c for c in (pref_col, "rating") if c not in df.columns]
        print(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        df[out_col] = (
            (np.tanh(df[pref_col] / scale) > tanh_thresh)
            & (df["rating"] > rating_thresh)
        ).astype("int8")
        return df

    return _add


def add_train_stats_to_test_df(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    mean_map = train_df.groupby("userId")["global_mean"].first()
    std_map = train_df.groupby("userId")["global_std"].first()

    test_df["global_mean"] = test_df["userId"].map(mean_map)
    test_df["global_std"] = test_df["userId"].map(std_map)
    return test_df


def calc_leave_one_out_metrics(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    l_approval = make_approval(scope="loo", scale=scale)
    g_approval = make_approval(scope="global", scale=scale)
    return (
        df.pipe(sum_ratings)
        .pipe(count_ratings)
        .pipe(sum_squares_ratings)
        .pipe(loo_mean)
        .pipe(loo_std)
        .pipe(loo_preference_direction)
        .pipe(l_approval)
        .pipe(global_mean)
        .pipe(global_std)
        .pipe(global_preference_direction)
        .pipe(g_approval)
    )


def calc_test_statistics(train_df: pd.DataFrame, test_df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    test_df = test_df.copy()
    test_df = add_train_stats_to_test_df(train_df, test_df)
    g_approval = make_approval(scope="global", scale=scale)
    return (test_df.pipe(global_preference_direction)
            .pipe(g_approval))


# return df
# 1. Calculate group sums, counts, and sum of squares
"""
    groups = df.groupby('userId')['rating']

    df['group_sum'] = groups.transform('sum')
    df['group_count'] = groups.transform('count')
    # sum of squares is needed for LOO standard deviation formula
    df['group_sum_sq'] = groups.transform(lambda x: (x**2).sum())

    # 2. Leave-One-Out Mean
    # Formula: (Total Sum - Current Value) / (Total Count - 1)
    df['loo_mean'] = (df['group_sum'] - df['rating']) / (df['group_count'] - 1)
    df['global_mean'] = df['group_sum'] / df['group_count']

    # 3. Leave-One-Out Standard Deviation
    # Formula for sample variance: (SumSq - (Sum^2 / Count)) / (Count - 1)
    # For LOO, we adjust Sum, SumSq, and Count by the current row
    loo_sum = df['group_sum'] - df['rating']
    loo_sum_sq = df['group_sum_sq'] - (df['rating']**2)
    loo_count = df['group_count'] - 1

    # Calculate variance then sqrt for std
    loo_var = (loo_sum_sq - (loo_sum**2 / loo_count)) / (loo_count - 1)
    # clip handles precision errors
    df['loo_std'] = np.sqrt(loo_var.clip(lower=0))
    df['global_std'] = np.sqrt(
        df['group_sum_sq'] / df['group_count'] - df['global_mean']**2)

    # applying tanh to the preference direction to bind it between [-1,1] for
    # approval estimation score anything above 1.0 std is considered approved
    df['preference_direction'] = (
        df['rating'] - df['loo_mean']) / df['loo_std']
    df['global_preference_direction'] = (
        df['rating'] - df['global_mean']) / df['global_std']
    # here we measure that each individual user has approved the movie by a threshold 1std and gave at least greater than 3 rating
    df['approval'] = (
        np.tanh(df['preference_direction']/scale) > 0.75).astype(int) * (df['rating'] > 3.0).astype(int)
    df['global_approval'] = (
        np.tanh(df['global_preference_direction']/scale) > 0.75).astype(int) * (df['rating'] > 3.0).astype(int)

    # Cleanup temporary columns
    df = df.drop(columns=['group_sum', 'group_count', 'group_sum_sq'])
    return df
"""
