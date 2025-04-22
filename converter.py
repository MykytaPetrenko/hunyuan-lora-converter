import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


# def decompose_lora_weight(big_matrix: torch.Tensor, rank: int = 32):
#     dtype = big_matrix.dtype
#     big_matrix = big_matrix.to(torch.float32)
#     U, S, Vh = torch.linalg.svd(big_matrix, full_matrices=False)
#     A = (U[:, :rank] * S[:rank].sqrt()).T
#     B = (S[:rank].sqrt()[:, None] * Vh[:rank])
#     return A.to(dtype).contiguous(), B.to(dtype).contiguous()

# def decompose_lora_weight(big_matrix: torch.Tensor, rank: int = 32):
#     dtype = big_matrix.dtype
#     device = big_matrix.device

#     big_matrix = big_matrix.to(torch.float32).cuda()
#     U, S, Vh = torch.linalg.svd(big_matrix, full_matrices=False)

#     A = (S[:rank].sqrt()[:, None] * Vh[:rank])         # (rank, in_dim)
#     B = (U[:, :rank] * S[:rank].sqrt()).to(big_matrix.dtype)  # (out_dim, rank)

#     return A.to(dtype).to(device).contiguous(), B.to(dtype).to(device).contiguous()

def decompose_lora_weight(big_matrix: torch.Tensor, rank: int = 32):
    dtype = big_matrix.dtype
    device = big_matrix.device
    
    big_matrix = big_matrix.to(torch.float32).cuda()
    U, S, Vh = torch.linalg.svd(big_matrix, full_matrices=False)
    
    # LoRA convention: A is down projection (rank, in_dim), B is up projection (out_dim, rank)
    A = (S[:rank].sqrt()[:, None] * Vh[:rank])  # (rank, in_dim)
    B = (U[:, :rank] * S[:rank].sqrt())  # (out_dim, rank)
    
    return A.to(dtype).to(device).contiguous(), B.to(dtype).to(device).contiguous()


def compute_accuracy(original, A, B):
    reconstructed = B @ A
    abs_diff = (original - reconstructed).abs().mean()
    base_mean = original.abs().mean()
    percent_error = 100 * abs_diff / base_mean
    return float(percent_error)


def convert_single_blocks(model, new_state):
    errors = []
    for i in tqdm(range(40), desc="Converting single_blocks"):
        prefix = f"lora_transformer_single_transformer_blocks_{i}"

        def mat(name): return model[f"{prefix}_{name}.lora_up.weight"] @ model[f"{prefix}_{name}.lora_down.weight"]

        q, k, v, mlp = mat("attn_to_q"), mat("attn_to_k"), mat("attn_to_v"), mat("proj_mlp")
        rank = model[f"{prefix}_proj_mlp.lora_up.weight"].shape[1]
        merged = torch.cat([q, k, v, mlp], dim=0)
        A1, B1 = decompose_lora_weight(merged, rank=rank)
        acc = compute_accuracy(merged, A1, B1)
        errors.append(acc)

        new_state[f"transformer.single_blocks.{i}.linear1.lora_A.weight"] = A1
        new_state[f"transformer.single_blocks.{i}.linear1.lora_B.weight"] = B1
        new_state[f"transformer.single_blocks.{i}.linear1.alpha"] = model[f"{prefix}_proj_mlp.alpha"]

        new_state[f"transformer.single_blocks.{i}.linear2.lora_A.weight"] = model[f"{prefix}_proj_out.lora_down.weight"]
        new_state[f"transformer.single_blocks.{i}.linear2.lora_B.weight"] = model[f"{prefix}_proj_out.lora_up.weight"]
        new_state[f"transformer.single_blocks.{i}.linear2.alpha"] = model[f"{prefix}_proj_out.alpha"]

        new_state[f"transformer.single_blocks.{i}.modulation.linear.lora_A.weight"] = model[f"{prefix}_norm_linear.lora_down.weight"]
        new_state[f"transformer.single_blocks.{i}.modulation.linear.lora_B.weight"] = model[f"{prefix}_norm_linear.lora_up.weight"]
        new_state[f"transformer.single_blocks.{i}.modulation.linear.alpha"] = model[f"{prefix}_norm_linear.alpha"]
    print(
        f"Single Block approx error:\n"
        f"average: {sum(errors)/len(errors):.4f}%\n"
        f"min: {min(errors):.4f}%\n"
        f"max: {max(errors):.4f}%\n"
    )

def convert_double_blocks(model, new_state):
    img_attn_qkv_errors = []
    txt_attn_qkv_errors = []
    for i in tqdm(range(20), desc="Converting double_blocks"):
        prefix = f"lora_transformer_transformer_blocks_{i}"

        def mat(name): return model[f"{prefix}_{name}.lora_up.weight"] @ model[f"{prefix}_{name}.lora_down.weight"]

        # --- img_attn_qkv ---
        q = mat("attn_to_q")
        k = mat("attn_to_k")
        v = mat("attn_to_v")
        rank = model[f"{prefix}_attn_to_q.lora_up.weight"].shape[1]
        merged_qkv = torch.cat([q, k, v], dim=0)
        A_qkv, B_qkv = decompose_lora_weight(merged_qkv, rank=rank)
        acc = compute_accuracy(merged_qkv, A_qkv, B_qkv)
        img_attn_qkv_errors.append(acc)
        new_state[f"transformer.double_blocks.{i}.img_attn_qkv.lora_A.weight"] = A_qkv
        new_state[f"transformer.double_blocks.{i}.img_attn_qkv.lora_B.weight"] = B_qkv
        new_state[f"transformer.double_blocks.{i}.img_attn_qkv.alpha"] = model[f"{prefix}_attn_to_q.alpha"]

        # --- txt_attn_qkv ---
        q = mat("attn_add_q_proj")
        k = mat("attn_add_k_proj")
        v = mat("attn_add_v_proj")
        rank = model[f"{prefix}_attn_add_q_proj.lora_up.weight"].shape[1]
        merged_qkv = torch.cat([q, k, v], dim=0)
        A_qkv, B_qkv = decompose_lora_weight(merged_qkv, rank=rank)
        acc = compute_accuracy(merged_qkv, A_qkv, B_qkv)
        txt_attn_qkv_errors.append(acc)
        new_state[f"transformer.double_blocks.{i}.txt_attn_qkv.lora_A.weight"] = A_qkv
        new_state[f"transformer.double_blocks.{i}.txt_attn_qkv.lora_B.weight"] = B_qkv
        new_state[f"transformer.double_blocks.{i}.txt_attn_qkv.alpha"] = model[f"{prefix}_attn_add_q_proj.alpha"]

        # --- img_attn_proj ---
        new_state[f"transformer.double_blocks.{i}.img_attn_proj.lora_A.weight"] = model[f"{prefix}_attn_to_out_0.lora_down.weight"]
        new_state[f"transformer.double_blocks.{i}.img_attn_proj.lora_B.weight"] = model[f"{prefix}_attn_to_out_0.lora_up.weight"]
        new_state[f"transformer.double_blocks.{i}.img_attn_proj.alpha"] = model[f"{prefix}_attn_to_out_0.alpha"]

        # --- txt_attn_proj ---
        new_state[f"transformer.double_blocks.{i}.txt_attn_proj.lora_A.weight"] = model[f"{prefix}_attn_to_add_out.lora_down.weight"]
        new_state[f"transformer.double_blocks.{i}.txt_attn_proj.lora_B.weight"] = model[f"{prefix}_attn_to_add_out.lora_up.weight"]
        new_state[f"transformer.double_blocks.{i}.txt_attn_proj.alpha"] = model[f"{prefix}_attn_to_add_out.alpha"]

        # --- mlp.fc1 ---
        new_state[f"transformer.double_blocks.{i}.img_mlp.fc1.lora_A.weight"] = model[f"{prefix}_ff_net_0_proj.lora_down.weight"]
        new_state[f"transformer.double_blocks.{i}.img_mlp.fc1.lora_B.weight"] = model[f"{prefix}_ff_net_0_proj.lora_up.weight"]
        new_state[f"transformer.double_blocks.{i}.img_mlp.fc1.alpha"] = model[f"{prefix}_ff_net_0_proj.alpha"]

        if f"{prefix}_ff_context_net_0_proj.lora_down.weight" in model:
            new_state[f"transformer.double_blocks.{i}.txt_mlp.fc1.lora_A.weight"] = model[f"{prefix}_ff_context_net_0_proj.lora_down.weight"]
            new_state[f"transformer.double_blocks.{i}.txt_mlp.fc1.lora_B.weight"] = model[f"{prefix}_ff_context_net_0_proj.lora_up.weight"]
            new_state[f"transformer.double_blocks.{i}.txt_mlp.fc1.alpha"] = model[f"{prefix}_ff_context_net_0_proj.alpha"]

        # --- mlp.fc2 ---
        new_state[f"transformer.double_blocks.{i}.img_mlp.fc2.lora_A.weight"] = model[f"{prefix}_ff_net_2.lora_down.weight"]
        new_state[f"transformer.double_blocks.{i}.img_mlp.fc2.lora_B.weight"] = model[f"{prefix}_ff_net_2.lora_up.weight"]
        new_state[f"transformer.double_blocks.{i}.img_mlp.fc2.alpha"] = model[f"{prefix}_ff_net_2.alpha"]

        if f"{prefix}_ff_context_net_2.lora_down.weight" in model:
            new_state[f"transformer.double_blocks.{i}.txt_mlp.fc2.lora_A.weight"] = model[f"{prefix}_ff_context_net_2.lora_down.weight"]
            new_state[f"transformer.double_blocks.{i}.txt_mlp.fc2.lora_B.weight"] = model[f"{prefix}_ff_context_net_2.lora_up.weight"]
            new_state[f"transformer.double_blocks.{i}.txt_mlp.fc2.alpha"] = model[f"{prefix}_ff_context_net_2.alpha"]
            

        # --- mod.linear ---
        if f"{prefix}_norm1_linear.lora_down.weight" in model:
            new_state[f"transformer.double_blocks.{i}.img_mod.linear.lora_A.weight"] = model[f"{prefix}_norm1_linear.lora_down.weight"]
            new_state[f"transformer.double_blocks.{i}.img_mod.linear.lora_B.weight"] = model[f"{prefix}_norm1_linear.lora_up.weight"]
            new_state[f"transformer.double_blocks.{i}.img_mod.linear.alpha"] = model[f"{prefix}_norm1_linear.alpha"]

        if f"{prefix}_norm1_context_linear.lora_down.weight" in model:
            new_state[f"transformer.double_blocks.{i}.txt_mod.linear.lora_A.weight"] = model[f"{prefix}_norm1_context_linear.lora_down.weight"]
            new_state[f"transformer.double_blocks.{i}.txt_mod.linear.lora_B.weight"] = model[f"{prefix}_norm1_context_linear.lora_up.weight"]
            new_state[f"transformer.double_blocks.{i}.txt_mod.linear.alpha"] = model[f"{prefix}_norm1_context_linear.alpha"]

    print(
        f"Double Block txt_attn_qkv approx error:\n"
        f"average: {sum(txt_attn_qkv_errors)/len(txt_attn_qkv_errors):.4f}%\n"
        f"min: {min(txt_attn_qkv_errors):.4f}%\n"
        f"max: {max(txt_attn_qkv_errors):.4f}%\n"
    )
    print(
        f"Double Block txt_attn_qkv approx error:\n"
        f"average: {sum(img_attn_qkv_errors)/len(img_attn_qkv_errors):.4f}%\n"
        f"min: {min(img_attn_qkv_errors):.4f}%\n"
        f"max: {max(img_attn_qkv_errors):.4f}%\n"
    )

def convert_all(in_path, out_path):
    model = load_file(in_path)
    new_state = {}

    convert_single_blocks(model, new_state)
    convert_double_blocks(model, new_state)

    save_file(new_state, out_path)
    print(f"✅ Saved merged model to {out_path}")


if __name__ == "__main__":
    convert_all("input_lora.safetensors", "converted_lora.safetensors")
