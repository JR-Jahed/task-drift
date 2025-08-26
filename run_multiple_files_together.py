from phi.run_multi_prompt_all_layers_attack import run_phi_attack_from_another_file
from llama.run_multi_prompt_all_layers_attack import run_llama_attack_from_another_file
from generate_activations.generate import generate_activations_from_another_file
import time


if __name__ == '__main__':

    wait_time = 10

    generate_activations_start = time.time()
    generate_activations_from_another_file()
    generate_activations_end = time.time()

    print(f"\n\n\n\n\nGenerating activations took {generate_activations_end - generate_activations_start} seconds...")
    time.sleep(wait_time)

    # ------------------------------------------------------------------------------------------------------------------------------

    phi_start = time.time()
    run_phi_attack_from_another_file()
    phi_end = time.time()

    print(f"\n\n\n\n\nPhi attack took {phi_end - phi_start} seconds...")
    time.sleep(wait_time)

    # ------------------------------------------------------------------------------------------------------------------------------

    llama_start = time.time()
    run_llama_attack_from_another_file()
    llama_end = time.time()

    print(f"\n\n\n\n\nLLaMA attack took {llama_end - llama_start} seconds...")
