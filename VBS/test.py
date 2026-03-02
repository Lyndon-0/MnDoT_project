from main import conservation_check

df = conservation_check(
    vbs_node_id="B23238",
    epsilon=2,
    network_path="/Users/parsa025/codes/estimating_erratic_errors/python_minimal/data/network.dat",
    flow_path="/Users/parsa025/codes/estimating_erratic_errors/python_minimal/data/flow_matrix.txt",
)
print(df.head(10))