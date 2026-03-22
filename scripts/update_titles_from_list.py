from pathlib import Path

import pandas as pd


TITLES = [
    "Analysis of Movie Intelligent Recommendation Algorithm based on MovieLens Platform",
    "Movie Recommendation Systems: Advanced Techniques for Personalized Content Suggestion",
    "Combining Cmab with Matrix Factorization and Clustering For Enhanced Movie Recommendations",
    "An Efficient Hybrid Parallel SGD Algorithm for Matrix Factorization in Recommendation Systems",
    "Hybrid Matrix Factorization Based Graph Contrastive Learning for Recommendation System",
    "Multi‑criteria recommendation system based on deep matrix factorization and regression techniques",
    "Implementation and Effectiveness Evaluation of Four Common Algorithms of Recommendation Systems: User Collaboration Filter, Item based Collaborative Filtering, Matrix Factorization and Neural Collaborative Filtering",
    "Neural Matrix Factorization Recommendation for User Preference Prediction Based on Explicit and Implicit Feedback",
    "Probabilistic Matrix Factorization Recommendation Approach for Integrating Multiple Information Sources",
    "TopC-CAMF: A Top Context Based Matrix Factorization Recommender System",
    "GAN-based Matrix Factorization for Recommender Systems",
    "Kernelized Deep Learning for Matrix Factorization Recommendation System Using Explicit and Implicit Information",
    "Comprehensive Evaluation of Matrix Factorization Models for Collaborative Filtering Recommender Systems",
    "Matrix Factorization Recommendation Algorithm Based on Attention Interaction",
    "Recommender Systems Based on Nonnegative Matrix Factorization: A Survey",
    "IMPROVED PROBABILISTIC MATRIX FACTORIZATION MODEL FOR SPARSE DATASETS",
    "Collaborative filtering recommendation algorithm based on user correlation and evolutionary clustering",
    "Sparse Matrix Factorization Algorithm for Real Time Movie Recommendation Systems",
    "A metaheuristic-based histogram equalization method for mammogram enhancement using a brightness preserving cuckoo search algorithm",
    "Gannet optimization algorithm : A new metaheuristic algorithm for solving engineering optimization problems",
    "Gradient-based optimizer: A new metaheuristic optimization algorithm",
    "Fire Hawk Optimizer: a novel metaheuristic algorithm",
    "State-of-the-Art Reviews of Meta-Heuristic Algorithms with Their Novel Proposal for Unconstrained Optimization and Applications",
    "Multi-Verse Optimizer: a nature-inspired algorithm for global optimization",
    "The Ant Lion Optimizer",
    "A new optimization method: Dolphin echolocation",
    "Gradient-based optimizer improved by Slime Mould Algorithm for global optimization and feature selection for diverse computation problems",
    "African vultures optimization algorithm: A new nature-inspired metaheuristic algorithm for global optimization problems",
    "Dung beetle optimizer: a new meta-heuristic algorithm for global optimization",
    "Improved Aquila optimizer and its applications",
    "Slime mould algorithm: a comprehensive review of recent variants and applications",
    "An Improved Equilibrium Optimizer Algorithm for Features Selection: Methods and Analysis",
    "A new optimization method: Big Bang–Big Crunch",
    "Emotion-Aware Movie Recommendation System using Hybrid Dipper-Throated and Grey Wolf Optimization with Autoencoder and Hybrid Filtering",
    "Hybrid crow search and uniform crossover algorithm-based clustering for top-N recommendation system",
    "Meta-Heuristic Algorithms for Learning Path Recommender at MOOC",
    "A survey of recommender systems with multi-objective optimization",
    "Applying particle swarm optimization algorithm-based collaborative filtering recommender system considering rating and review",
    "A collaborative recommender system enhanced with particle swarm optimization technique",
    "Recommender system with grey wolf optimizer and FCM",
    "A new recommendation system using map-reduce-based tournament empowered Whale optimization algorithm",
    "A grasshopper optimization algorithm‑based movie recommender system",
    "A Movie Recommender System Based on User Profile and Artificial Bee Colony Optimization",
    "Personalized E-commerce Recommendation System using Marine Predator Algorithm with Gated Recurrent Unit",
    "A recommender system with multi-objective hybrid Harris Hawk optimization for feature selection and disease diagnosis",
    "Movie recommender system with metaheuristic artificial bee",
    "Hybrid Sparrow Clustered (HSC) Algorithm for Top-N Recommendation System",
    "Proposing improved meta‑heuristic algorithms for clustering and separating users in the recommender systems",
]


def main() -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    xlsx_path = docs_dir / "literature.xlsx"
    if not xlsx_path.exists():
        raise SystemExit(f"Excel not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    first_col = df.columns[0]
    n = min(len(df), len(TITLES))
    df.loc[: n - 1, first_col] = TITLES[:n]

    df.to_excel(xlsx_path, index=False)
    print(f"Updated first column '{first_col}' for first {n} rows (rows={len(df)}, titles={len(TITLES)}).")


if __name__ == "__main__":
    main()

