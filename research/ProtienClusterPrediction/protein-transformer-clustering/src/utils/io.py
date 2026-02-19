def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()

def save_cluster_results(cluster_results, filepath):
    with open(filepath, 'w') as f:
        for cluster_id, data in cluster_results.items():
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"  Count: {data['count']}\n")
            f.write(f"  GC Content: {data['gc']}\n")
            f.write(f"  Isolates: {dict(data['isolates'])}\n")
            f.write("\n")

def load_cluster_results(filepath):
    cluster_results = {}
    with open(filepath, 'r') as f:
        current_cluster = None
        for line in f:
            if line.startswith("Cluster"):
                current_cluster = int(line.split()[1])
                cluster_results[current_cluster] = {'count': 0, 'gc': [], 'isolates': Counter()}
            elif line.startswith("  Count:"):
                cluster_results[current_cluster]['count'] = int(line.split()[1])
            elif line.startswith("  GC Content:"):
                cluster_results[current_cluster]['gc'] = float(line.split()[2])
            elif line.startswith("  Isolates:"):
                isolates = eval(line.split(":")[1].strip())
                cluster_results[current_cluster]['isolates'] = Counter(isolates)
    return cluster_results