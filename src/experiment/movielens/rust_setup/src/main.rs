use anyhow::{bail, Context, Result};
use clap::Parser;
use rand::rngs::StdRng;
use rand::seq::index;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DATASET_VERSION: &str = "ml-32m";
const MIN_REVIEWS: usize = 1000;
const MIN_INDIVIDUAL_USER_REVIEWS: usize = 200;

#[derive(Parser, Debug)]
#[command(name = "movielens_setup_rust")]
#[command(about = "MovieLens setup JSONL generator (Rust)")]
struct Cli {
    #[arg(long = "setup", value_name = "FILENAME")]
    setup: String,
    #[arg(long = "num-users", default_value_t = 5)]
    num_users: usize,
    #[arg(long = "num-runs", default_value_t = 10_000)]
    num_runs: usize,
    #[arg(long = "valid-movie-count", default_value_t = 50)]
    valid_movie_count: usize,
    #[arg(long = "seed", default_value_t = 43)]
    seed: i64,
}

#[derive(Debug, Deserialize)]
struct RatingRow {
    #[serde(rename = "userId")]
    user_id: u32,
    #[serde(rename = "movieId")]
    movie_id: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct UserMovies {
    user_id: u32,
    movie_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct SampleResult {
    movie_ids: Vec<u32>,
    user_ids: Vec<u32>,
    valid_movie_count: usize,
}

#[derive(Debug, Serialize)]
struct SetupConfig {
    seed: i64,
    num_runs: usize,
    valid_movie_count: usize,
    num_users: usize,
    output_file: String,
    exp_valid_movie_count: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct FileSig {
    len: u64,
    modified_unix_secs: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheBlob {
    ratings_sig: FileSig,
    movies_sig: FileSig,
    min_reviews: usize,
    min_individual_user_reviews: usize,
    users: Vec<UserMovies>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli)
}

fn run(cli: Cli) -> Result<()> {
    let repo_root = find_repo_root(&std::env::current_dir()?)?;
    let dataset_root = repo_root.join(".dataset").join(DATASET_VERSION);
    let movies_path = dataset_root.join("movies.csv");
    let ratings_path = dataset_root.join("ratings.csv");
    ensure_exists(&movies_path)?;
    ensure_exists(&ratings_path)?;

    let movie_count = count_movies_rows(&movies_path)?;
    if movie_count == 0 {
        bail!("movies.csv exists but contains no data rows");
    }

    let cache_path = repo_root
        .join(".dataset")
        .join(".cache")
        .join("indices_ml-32m_minrev1000_minind200_rust.bin");
    let users = load_or_build_users_cache(&ratings_path, &movies_path, &cache_path)?;

    if cli.num_users == 0 {
        bail!("--num-users must be > 0");
    }
    if cli.num_users > users.len() {
        bail!(
            "--num-users ({}) exceeds available filtered users ({})",
            cli.num_users,
            users.len()
        );
    }

    let mut rng = StdRng::seed_from_u64(cli.seed as u64);
    let mut samples: Vec<SampleResult> = Vec::new();

    for _ in 0..cli.num_runs {
        let picks = index::sample(&mut rng, users.len(), cli.num_users).into_vec();
        let mut selected: Vec<&UserMovies> = picks.iter().map(|i| &users[*i]).collect();
        selected.sort_unstable_by_key(|u| u.user_id);

        let valid_movie_ids = intersect_all(&selected);
        let valid_count = valid_movie_ids.len();
        if valid_count > cli.valid_movie_count {
            samples.push(SampleResult {
                movie_ids: valid_movie_ids,
                user_ids: selected.iter().map(|u| u.user_id).collect(),
                valid_movie_count: valid_count,
            });
        }
    }

    samples.sort_by_key(|s| {
        (
            Reverse(s.valid_movie_count),
            s.movie_ids.len(),
            s.user_ids.len(),
            s.movie_ids.first().copied().unwrap_or_default(),
            s.user_ids.first().copied().unwrap_or_default(),
        )
    });

    let output_path = PathBuf::from(&cli.setup);
    let config_path = derive_config_path(&output_path);
    write_jsonl_samples_atomic(&samples, &output_path)?;

    let config = SetupConfig {
        seed: cli.seed,
        num_runs: cli.num_runs,
        valid_movie_count: cli.valid_movie_count,
        num_users: cli.num_users,
        output_file: cli.setup,
        exp_valid_movie_count: 100,
    };
    write_json_atomic(&config, &config_path)?;

    println!("Wrote samples to: {}", output_path.display());
    println!("Wrote config to: {}", config_path.display());
    Ok(())
}

fn derive_config_path(output_path: &Path) -> PathBuf {
    let parent = output_path.parent().unwrap_or_else(|| Path::new(""));
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("setup");
    parent.join(format!("{stem}_config.json"))
}

fn load_or_build_users_cache(
    ratings_path: &Path,
    movies_path: &Path,
    cache_path: &Path,
) -> Result<Vec<UserMovies>> {
    let ratings_sig = file_signature(ratings_path)?;
    let movies_sig = file_signature(movies_path)?;
    if let Ok(raw) = fs::read(cache_path) {
        if let Ok(blob) = bincode::deserialize::<CacheBlob>(&raw) {
            if blob.ratings_sig == ratings_sig
                && blob.movies_sig == movies_sig
                && blob.min_reviews == MIN_REVIEWS
                && blob.min_individual_user_reviews == MIN_INDIVIDUAL_USER_REVIEWS
            {
                return Ok(blob.users);
            }
        }
    }

    let users = build_user_movie_index(ratings_path)?;
    let blob = CacheBlob {
        ratings_sig,
        movies_sig,
        min_reviews: MIN_REVIEWS,
        min_individual_user_reviews: MIN_INDIVIDUAL_USER_REVIEWS,
        users: users.clone(),
    };
    let payload = bincode::serialize(&blob).context("failed to serialize cache")?;
    write_bytes_atomic(&payload, cache_path)?;
    Ok(users)
}

fn build_user_movie_index(ratings_path: &Path) -> Result<Vec<UserMovies>> {
    let mut movie_counts: HashMap<u32, usize> = HashMap::new();
    {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(ratings_path)
            .with_context(|| format!("failed to open {}", ratings_path.display()))?;
        for row in reader.deserialize::<RatingRow>() {
            let row = row.context("invalid ratings.csv row while counting movies")?;
            *movie_counts.entry(row.movie_id).or_insert(0) += 1;
        }
    }

    let valid_movies: HashSet<u32> = movie_counts
        .into_iter()
        .filter_map(|(mid, n)| (n >= MIN_REVIEWS).then_some(mid))
        .collect();

    let mut user_counts: HashMap<u32, usize> = HashMap::new();
    let mut movie_filtered_pairs: Vec<(u32, u32)> = Vec::new();
    {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(ratings_path)
            .with_context(|| format!("failed to open {}", ratings_path.display()))?;
        for row in reader.deserialize::<RatingRow>() {
            let row = row.context("invalid ratings.csv row while counting users")?;
            if valid_movies.contains(&row.movie_id) {
                movie_filtered_pairs.push((row.user_id, row.movie_id));
                *user_counts.entry(row.user_id).or_insert(0) += 1;
            }
        }
    }

    let valid_users: HashSet<u32> = user_counts
        .into_iter()
        .filter_map(|(uid, n)| (n >= MIN_INDIVIDUAL_USER_REVIEWS).then_some(uid))
        .collect();

    let mut by_user: HashMap<u32, Vec<u32>> = HashMap::new();
    for (uid, mid) in movie_filtered_pairs {
        if valid_users.contains(&uid) {
            by_user.entry(uid).or_default().push(mid);
        }
    }

    let mut users: Vec<UserMovies> = by_user
        .into_iter()
        .map(|(user_id, mut movie_ids)| {
            movie_ids.sort_unstable();
            movie_ids.dedup();
            UserMovies { user_id, movie_ids }
        })
        .collect();
    users.sort_unstable_by_key(|u| u.user_id);
    Ok(users)
}

fn intersect_all(selected: &[&UserMovies]) -> Vec<u32> {
    if selected.is_empty() {
        return Vec::new();
    }
    let mut acc = selected[0].movie_ids.clone();
    for user_movies in &selected[1..] {
        acc = intersect_sorted(&acc, &user_movies.movie_ids);
        if acc.is_empty() {
            break;
        }
    }
    acc
}

fn intersect_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

fn file_signature(path: &Path) -> Result<FileSig> {
    let meta = fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    let modified = meta
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Ok(FileSig {
        len: meta.len(),
        modified_unix_secs: modified,
    })
}

fn count_movies_rows(path: &Path) -> Result<usize> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let mut count = 0usize;
    for record in reader.records() {
        record.context("invalid movies.csv row")?;
        count += 1;
    }
    Ok(count)
}

fn ensure_exists(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("required file not found: {}", path.display());
    }
    Ok(())
}

fn write_jsonl_samples_atomic(samples: &[SampleResult], path: &Path) -> Result<()> {
    let mut buf = Vec::new();
    for sample in samples {
        serde_json::to_writer(&mut buf, sample).context("failed to encode sample JSON")?;
        buf.push(b'\n');
    }
    write_bytes_atomic(&buf, path)
}

fn write_json_atomic<T: Serialize>(value: &T, path: &Path) -> Result<()> {
    let mut buf = serde_json::to_vec_pretty(value).context("failed to encode config JSON")?;
    buf.push(b'\n');
    write_bytes_atomic(&buf, path)
}

fn write_bytes_atomic(bytes: &[u8], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let tmp = path.with_extension(format!(
        "tmp.{}.{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    {
        let mut file =
            fs::File::create(&tmp).with_context(|| format!("failed to create {}", tmp.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("failed to write {}", tmp.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to flush {}", tmp.display()))?;
    }
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "failed to atomically move {} -> {}",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn find_repo_root(start: &Path) -> Result<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        let has_git = current.join(".git").exists();
        let has_pyproject = current.join("pyproject.toml").exists();
        if has_git || has_pyproject {
            return Ok(current);
        }
        if !current.pop() {
            bail!(
                "could not locate repo root from {} (missing .git/pyproject.toml)",
                start.display()
            );
        }
    }
}
