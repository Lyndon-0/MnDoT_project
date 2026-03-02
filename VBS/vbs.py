import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.stderr = sys.stdout


def load_flow_matrix(path: str | Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


def load_incidence_matrix(
    network_path: str | Path,
    remove_od_nodes: bool = True,
    return_node_ids: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    network_path = Path(network_path)
    delimiter = "," if network_path.suffix.lower() == ".csv" else None
    network = np.loadtxt(network_path, skiprows=1, usecols=(0, 1), dtype=str, delimiter=delimiter)
    network = np.atleast_2d(network)

    link_count = network.shape[0]
    node_to_idx: dict[str, int] = {}
    for start_node, end_node in network:
        if start_node not in node_to_idx:
            node_to_idx[start_node] = len(node_to_idx)
        if end_node not in node_to_idx:
            node_to_idx[end_node] = len(node_to_idx)

    node_ids = sorted(node_to_idx, key=lambda k: node_to_idx[k])
    matrix = np.zeros((len(node_to_idx), link_count), dtype=float)
    for link_id, (start_node, end_node) in enumerate(network):
        matrix[node_to_idx[start_node], link_id] = -1.0
        matrix[node_to_idx[end_node], link_id] = 1.0

    # Keep MATLAB's fixed OD-node row removal only for the original 19-node benchmark.
    if remove_od_nodes:
        numeric_labels = [label for label in node_to_idx.keys() if label.isdigit()]
        if len(numeric_labels) == len(node_to_idx):
            numeric_values = sorted(int(label) for label in numeric_labels)
            is_original_layout = numeric_values == list(range(1, 20))
            if is_original_layout:
                matrix = np.delete(matrix, np.s_[13:19], axis=0)
                node_ids = [node_id for idx, node_id in enumerate(node_ids) if idx < 13 or idx >= 19]

    if return_node_ids:
        return matrix, node_ids
    return matrix


def conservation_check(
    vbs_node_id: str | int,
    epsilon: float = 1e-6,
    network_path: str | Path | None = None,
    flow_path: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(__file__).resolve().parent
    if network_path is None:
        network_path = root / "data" / "network.dat"
    if flow_path is None:
        flow_path = root / "data" / "flow_matrix.txt"

    network_path = Path(network_path)
    flow_path = Path(flow_path)

    network_delimiter = "," if network_path.suffix.lower() == ".csv" else None
    network = np.loadtxt(network_path, skiprows=1, usecols=(0, 1), dtype=str, delimiter=network_delimiter)
    network = np.atleast_2d(network)

    flow_delimiter = "," if flow_path.suffix.lower() == ".csv" else None
    flow_matrix = np.loadtxt(flow_path, dtype=float, delimiter=flow_delimiter)
    flow_matrix = np.atleast_2d(flow_matrix)

    if flow_matrix.shape[0] != network.shape[0]:
        raise ValueError(
            f"flow_matrix rows ({flow_matrix.shape[0]}) must match number of edges in network ({network.shape[0]})."
        )

    node_key = str(vbs_node_id)
    incoming_edges = np.where(network[:, 1] == node_key)[0]
    outgoing_edges = np.where(network[:, 0] == node_key)[0]

    if incoming_edges.size > 0:
        incoming_sum = flow_matrix[incoming_edges, :].sum(axis=0)
    else:
        incoming_sum = np.zeros(flow_matrix.shape[1], dtype=float)

    if outgoing_edges.size > 0:
        outgoing_sum = flow_matrix[outgoing_edges, :].sum(axis=0)
    else:
        outgoing_sum = np.zeros(flow_matrix.shape[1], dtype=float)

    abs_diff = np.abs(incoming_sum - outgoing_sum)
    conservation_flag = (abs_diff > epsilon).astype(int)

    return pd.DataFrame({"conservation_flag": conservation_flag})


def obtain_error_probability(
    A: np.ndarray,
    rng: np.random.Generator,
    network_path: str | Path,
    flow_path: str | Path,
    epsilon: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    network_path = Path(network_path)
    delimiter = "," if network_path.suffix.lower() == ".csv" else None
    raw_network = np.loadtxt(network_path, skiprows=1, dtype=str, delimiter=delimiter)
    raw_network = np.atleast_2d(raw_network)

    edge_nodes = raw_network[:, :2]
    link_features = raw_network[:, 2:].astype(float)

    if link_features.shape[0] != A.shape[1]:
        raise ValueError(
            f"network edges ({link_features.shape[0]}) must match A link dimension ({A.shape[1]})."
        )

    _, node_ids = load_incidence_matrix(network_path, remove_od_nodes=True, return_node_ids=True)
    node_ids = np.array(node_ids, dtype=str)

    vbs_mask = np.array([node_id.startswith("B") for node_id in node_ids], dtype=bool)
    if np.any(vbs_mask):
        selected_rows = np.where(vbs_mask)[0]
    else:
        selected_rows = np.arange(len(node_ids))

    p_node_selected = np.zeros(selected_rows.size, dtype=float)
    for local_idx, row_idx in enumerate(selected_rows):
        conv_flags = conservation_check(
            vbs_node_id=node_ids[row_idx],
            epsilon=epsilon,
            network_path=network_path,
            flow_path=flow_path,
        )
        p_node_selected[local_idx] = float(conv_flags["conservation_flag"].mean())

    p_node_selected = np.clip(p_node_selected, 0.0, 1 - 1e-5)
    node_y = -np.log(1 - p_node_selected)
    node_X = np.abs(A[selected_rows, :]) @ link_features
    b, _, _, _ = np.linalg.lstsq(node_X, node_y, rcond=None)
    estimated_p_link = 1 - np.exp(-(link_features @ b))
    estimated_p_link = np.clip(estimated_p_link, 0.01, 0.99)

    p_node = np.zeros(A.shape[0], dtype=float)
    p_node[selected_rows] = p_node_selected

    sensor_prob_lists: dict[str, list[float]] = {}
    for edge_idx, (start_node, end_node) in enumerate(edge_nodes):
        start_is_vbs = start_node.startswith("B")
        end_is_vbs = end_node.startswith("B")
        if start_is_vbs ^ end_is_vbs:
            sensor_id = str(end_node if start_is_vbs else start_node)
            sensor_prob_lists.setdefault(sensor_id, []).append(float(estimated_p_link[edge_idx]))

    sensor_probabilities = {str(sensor_id): float(np.mean(values)) for sensor_id, values in sensor_prob_lists.items()}

    return estimated_p_link, p_node, link_features, sensor_probabilities


def _sample_noise(sigma: float, rng: np.random.Generator) -> float:
    rand_dis = rng.random()
    if rand_dis < 0.25:
        return float(np.round(rng.normal(0.0, sigma)))
    if rand_dis < 0.5:
        return float(np.round((2 * rng.random() - 1) * sigma))
    if rand_dis < 0.75:
        return float(np.round(rng.exponential(sigma)))
    return float(np.round(rng.gamma(shape=np.sqrt(sigma), scale=np.sqrt(sigma))))


def generate_error_data(true_data: np.ndarray, p_link: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    link_number, time_interval_number = true_data.shape
    wrong_link_number = np.round(p_link * time_interval_number).astype(int)

    error_data = true_data.copy()
    for i in range(link_number):
        n_wrong = int(wrong_link_number[i])
        if n_wrong <= 0:
            continue
        wrong_times = rng.permutation(time_interval_number)[:n_wrong]
        for t in wrong_times:
            error_data[i, t] = true_data[i, t] + _sample_noise(sigma, rng)
            while error_data[i, t] < 0 or error_data[i, t] == true_data[i, t]:
                error_data[i, t] = true_data[i, t] / 2 + np.round(rng.normal(0.0, sigma))

    return error_data


def generate_error_node_flow(
    true_data: np.ndarray,
    p_node: np.ndarray,
    sigma: float,
    A: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    node_number, _ = A.shape
    _, time_interval_number = true_data.shape
    error_node_flow = A @ true_data

    wrong_number = np.round(p_node * time_interval_number).astype(int)
    for i in range(node_number):
        n_wrong = int(wrong_number[i])
        if n_wrong <= 0:
            continue
        wrong_time_index = rng.permutation(time_interval_number)[:n_wrong]
        for t in wrong_time_index:
            error_node_flow[i, t] = error_node_flow[i, t] + _sample_noise(sigma, rng)

    return error_node_flow


def calculate_p_from_data(
    A: np.ndarray,
    error_data: np.ndarray,
    error_node_flow: np.ndarray,
    X: np.ndarray,
    true_p_node: np.ndarray,
    true_p_link: np.ndarray,
    conservation_result: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_interval_number = error_data.shape[1]
    flow_diff = (error_node_flow != conservation_result).sum(axis=1)
    estimated_p_node = flow_diff / time_interval_number

    mape_p_node = np.mean(np.abs(estimated_p_node - true_p_node) / true_p_node)
    print("mape_p_node =")
    print()
    print(float(mape_p_node))

    true_p_node_noisy = true_p_node + 0.001 * rng.normal(size=true_p_node.shape)
    true_p_node_noisy[true_p_node_noisy >= 1] = 1 - 1e-5
    print()
    print(true_p_node_noisy)

    node_y = -np.log(1 - true_p_node_noisy)
    node_X = np.abs(A) @ X
    b, _, _, _ = np.linalg.lstsq(node_X, node_y, rcond=None)
    print()
    print(b)

    estimated_p_link = 1 - np.exp(-(X @ b))
    estimated_p_link = np.clip(estimated_p_link, 0.01, 0.99)

    mape_p_link = np.mean(np.abs(estimated_p_link - true_p_link) / true_p_link)
    print()
    print("mape_p_link_from_data")
    print(f"    {mape_p_link:.4f}")

    return estimated_p_link, b, estimated_p_node


def get_link_conservation_flag(A: np.ndarray, error_data: np.ndarray, conservation_result: np.ndarray) -> np.ndarray:
    observed_flow_conservation = A @ error_data
    node_conservation_flag = observed_flow_conservation == conservation_result

    _, time_interval_number = node_conservation_flag.shape
    node_number, link_number = A.shape
    link_conservation_flag = np.zeros((link_number, time_interval_number), dtype=float)

    for i in range(node_number):
        for t in range(time_interval_number):
            if node_conservation_flag[i, t]:
                temp_flag = np.abs(A[i, :])
                link_conservation_flag[:, t] = temp_flag

    return link_conservation_flag


def feasible_solution_exists(
    A: np.ndarray,
    error_link_index_origin: np.ndarray,
    node_conservation_flag: np.ndarray,
) -> bool:
    link_number, time_interval_number = error_link_index_origin.shape

    for link_index in range(link_number):
        for t in range(time_interval_number):
            if error_link_index_origin[link_index, t] == 1:
                connected_node = np.where(A[:, link_index] != 0)[0]
                if np.sum(node_conservation_flag[connected_node, t]) > 0:
                    return False

    return True


def _soft_threshold(tau: float, Q: np.ndarray) -> np.ndarray:
    return np.sign(Q) * np.maximum(np.abs(Q) - tau, 0.0)


def _svd_threshold(tau: float, Q: np.ndarray) -> np.ndarray:
    U, s, Vt = np.linalg.svd(Q, full_matrices=False)
    s_shrink = np.maximum(s - tau, 0.0)
    return (U * s_shrink) @ Vt


def our_admm(Q: np.ndarray, max_steps: int = 1000, error_tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    unobserved = np.isnan(Q)
    Q = Q.copy()
    Q[unobserved] = 0.0

    M, N = Q.shape
    normQ = np.linalg.norm(Q, ord="fro")
    rank_reg = 1 / np.sqrt(max(M, N))
    mu = 10 * rank_reg

    c_gamma = np.zeros((M, N), dtype=float)
    c_error = np.zeros((M, N), dtype=float)
    Y = np.zeros((M, N), dtype=float)

    for step in range(1, max_steps + 1):
        c_gamma = _svd_threshold(1 / mu, Q - c_error + (1 / mu) * Y)
        c_error = _soft_threshold(rank_reg / mu, Q - c_gamma + (1 / mu) * Y)

        Z = Q - c_gamma - c_error
        Z[unobserved] = 0.0
        Y = Y + mu * Z

        err = np.linalg.norm(Z, ord="fro") / normQ
        if step == 1 or step % 10 == 0 or err < error_tol:
            rank_gamma = np.linalg.matrix_rank(c_gamma)
            card_error = int(np.count_nonzero(c_error[~unobserved]))
            print(
                f"step: {step:04d}\terr: {err:.6f}\trank(c_gamma): {rank_gamma}\tcard(c_error): {card_error}"
            )
        if err < error_tol:
            break

    return c_gamma, c_error


def build_feasible_gamma(
    A: np.ndarray,
    error_data: np.ndarray,
    p_link: np.ndarray,
    node_conservation_flag: np.ndarray,
    v_gamma: np.ndarray,
    rng: np.random.Generator,
    pre_rank: int = 3,
) -> np.ndarray:
    node_number, link_number = A.shape
    time_interval_number = error_data.shape[1]

    delta = 1 - node_conservation_flag
    LM = np.round(p_link * time_interval_number).astype(int)

    zero_list = np.zeros((link_number, time_interval_number), dtype=float)
    fes_gamma = np.zeros((link_number, time_interval_number), dtype=float)

    for i in range(node_number):
        for t in range(time_interval_number):
            if delta[i, t] == 0:
                link_index_array = np.where(A[i, :] != 0)[0]
                for link_index in link_index_array:
                    zero_list[link_index, t] = 1

    gamma_index = np.argsort(-v_gamma, axis=1)
    for j in range(link_number):
        number_of_e_data = int(min(LM[j], time_interval_number - pre_rank))
        if number_of_e_data <= 0:
            continue
        time_index = gamma_index[j, :number_of_e_data]
        for t in time_index:
            if zero_list[j, t] == 0:
                fes_gamma[j, t] = 1

    for i in range(node_number):
        for t in range(time_interval_number):
            if delta[i, t] == 1:
                link_index_array = np.where(A[i, :] != 0)[0]
                selected_link_id = int(rng.choice(link_index_array))
                while zero_list[selected_link_id, t] == 1:
                    selected_link_id = int(rng.choice(link_index_array))
                fes_gamma[selected_link_id, t] = 1

    return fes_gamma


def recover_flow(
    error_data: np.ndarray,
    A: np.ndarray,
    true_flow_matrix: np.ndarray,
    fes_gamma: np.ndarray,
    big_number: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    import cvxpy as cp

    link_number, time_interval_number = error_data.shape
    actual_conservation_result = A @ true_flow_matrix

    sub_Q = cp.Variable((link_number, time_interval_number))
    sub_Z = cp.Variable((link_number, time_interval_number))

    constraints = [
        sub_Q == np.multiply(1 - fes_gamma, error_data) + sub_Z,
        sub_Z - np.multiply(fes_gamma, big_number) <= 0,
        sub_Q >= 0,
        sub_Z >= 0,
    ]

    # MATLAB uses norm(X,1). For matrices this is the induced 1-norm (max column sum),
    # so use cp.norm(..., 1) instead of entrywise cp.norm1(...).
    objective = cp.Minimize(cp.normNuc(sub_Q) + 100000 * cp.norm(A @ sub_Q - actual_conservation_result, 1))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)

    return sub_Q.value, sub_Z.value


def run(seed: int = 10, flow_path: str | None = None, network_path: str | None = None) -> dict[str, float]:
    # Use MT19937-based RandomState for behavior closer to MATLAB rng.
    rng = np.random.RandomState(seed)
    root = Path(__file__).resolve().parents[1]

    if flow_path is None:
        flow_path = str(root / "flow_matrix.txt")
    if network_path is None:
        network_path = str(root / "network.dat")

    true_flow_matrix = load_flow_matrix(flow_path)
    A = load_incidence_matrix(network_path, remove_od_nodes=True)
    conservation_result = A @ true_flow_matrix

    p_link, p_node, X, sensor_probabilities = obtain_error_probability(
        A,
        rng,
        network_path=network_path,
        flow_path=flow_path,
        epsilon=2.0,
    )
    print("sensor_probabilities:")
    print(sensor_probabilities)
    return sensor_probabilities
    sigma = float(np.std(true_flow_matrix))

    error_data = generate_error_data(true_flow_matrix, p_link, sigma, rng)
    error_node_flow = generate_error_node_flow(true_flow_matrix, p_node, sigma, A, rng)

    ae_origin = np.abs(error_data - true_flow_matrix)
    mape_origin = float(np.mean(ae_origin / (true_flow_matrix + 1)))
    error_link_index_origin = (ae_origin != 0).astype(int)

    estimated_p_link, estimated_b, estimated_p_node = calculate_p_from_data(
        A,
        error_data,
        error_node_flow,
        X,
        p_node,
        p_link,
        conservation_result,
        rng,
    )
    _ = get_link_conservation_flag(A, error_data, conservation_result)

    observed_conservation_result = A @ error_data
    node_conservation_flag = (observed_conservation_result == conservation_result).astype(int)

    while not feasible_solution_exists(A, error_link_index_origin, node_conservation_flag):
        error_data = generate_error_data(true_flow_matrix, p_link, sigma, rng)
        error_node_flow = generate_error_node_flow(true_flow_matrix, p_node, sigma, A, rng)

        ae_origin = np.abs(error_data - true_flow_matrix)
        mape_origin = float(np.mean(ae_origin / (true_flow_matrix + 1)))
        error_link_index_origin = (ae_origin != 0).astype(int)

        estimated_p_link, estimated_b, estimated_p_node = calculate_p_from_data(
            A,
            error_data,
            error_node_flow,
            X,
            p_node,
            p_link,
            conservation_result,
            rng,
        )

        _ = get_link_conservation_flag(A, error_data, conservation_result)

        observed_conservation_result = A @ error_data
        node_conservation_flag = (observed_conservation_result == conservation_result).astype(int)

    print("estimated_p_node_per_node:")
    for node_idx, p_err in enumerate(estimated_p_node, start=1):
        print(f"node_{node_idx}: {p_err:.6f}")

    _, S = our_admm(error_data)
    # Match MATLAB behavior: abs(S)./error_data (can produce inf when denominator is zero).
    v_gamma = np.abs(S) / error_data

    fes_gamma = build_feasible_gamma(
        A=A,
        error_data=error_data,
        p_link=p_link,
        node_conservation_flag=node_conservation_flag,
        v_gamma=v_gamma,
        rng=rng,
        pre_rank=3,
    )

    v_Q, _ = recover_flow(
        error_data=error_data,
        A=A,
        true_flow_matrix=true_flow_matrix,
        fes_gamma=fes_gamma,
        big_number=200.0,
    )

    sub_ae_result = np.abs(v_Q - true_flow_matrix)
    sub_mape_result = float(np.mean(sub_ae_result / (true_flow_matrix + 1)))

    print(f"origin_mape: {mape_origin:.5f}")
    print(f"our_mape: {sub_mape_result:.8f}")

    return {"origin_mape": mape_origin, "our_mape": sub_mape_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--flow-path", type=str, default=None)
    parser.add_argument("--network-path", type=str, default=None)
    args = parser.parse_args()

    run(seed=args.seed, flow_path=args.flow_path, network_path=args.network_path)
