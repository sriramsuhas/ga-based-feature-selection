import numpy as np
from sklearn.ensemble import RandomForestClassifier

def run_ga(X, y, pop_size=40, n_gens=25, mutation_rate=0.02, crossover_rate=0.8):
    n_features = X.shape[1]
    population = [np.random.choice([0, 1], size=n_features, p=[0.1, 0.9]).tolist() for _ in range(pop_size)]
    best_chromosome, best_train_acc = None, 0.0

    print("[GA Selection] Running GA for feature selection...")
    for gen in range(n_gens):
        gen_train_accs = []
        for chrom in population:
            selected = np.where(np.array(chrom) == 1)[0]
            if len(selected) == 0:
                gen_train_accs.append(0.0)
            else:
                X_train = X[:, selected]
                clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                clf.fit(X_train, y)
                train_acc = clf.score(X_train, y)
                gen_train_accs.append(train_acc)
        gen_best_idx = int(np.argmax(gen_train_accs))
        if gen_train_accs[gen_best_idx] > best_train_acc:
            best_train_acc = gen_train_accs[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()
        print(f"[GA] Gen {gen+1}/{n_gens} best train acc={best_train_acc:.4f} feat={np.sum(best_chromosome)}")
        new_pop = [population[gen_best_idx]]
        while len(new_pop) < pop_size:
            parents = np.random.choice(pop_size, size=2, replace=False)
            p1, p2 = population[parents[0]], population[parents[1]]
            pt = np.random.randint(1, n_features)
            child1 = p1[:pt] + p2[pt:]
            child2 = p2[:pt] + p1[pt:]
            for child in [child1, child2]:
                for i in range(n_features):
                    if np.random.rand() < mutation_rate:
                        child[i] = 1 - child[i]
                if np.sum(child) == 0:
                    child[np.random.randint(n_features)] = 1
                new_pop.append(child)
            if len(new_pop) >= pop_size:
                break
        population = new_pop[:pop_size]
    print("[GA Selection] Done! Best mask returned, best train accuracy:", best_train_acc)
    return np.array(best_chromosome)

def apply_feature_mask(npz_file, out_file, mask):
    d = np.load(npz_file)
    X, y = d["X"], d["y"]
    selected = mask.astype(bool)
    np.savez(out_file, X=X[:, selected], y=y)
