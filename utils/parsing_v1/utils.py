import os

def prepare_output_dir(base_dir, doc_id):
    out_dir = os.path.join(base_dir, doc_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
