#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo "$(date), Setup the environment ..."
set -e  # exit on error

# -------------------------
# paths
# -------------------------
exp_path=exp_location_multi_cp
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

# -------------------------
# common configs
# -------------------------
task="rewrite"
# Ms=("gemma-2b-instruct" "qwen-4b-instruct" "mistralai-8b-instruct" "gpt-4o")
Ms=("gemma-2b-instruct")
datasets=("xsum" "writing")
datasets=("writing")
phis=("FDGPT" "FT")
cp_methods=("NOT")
power_list=(1.0)
ncp=3

# preparing dataset
for M in "${Ms[@]}"; do
    for D in "${datasets[@]}"; do
      echo "$(date), Preparing dataset ${D}_${M}_${task} ..."

      python scripts/data_builder_sentence.py \
        --dataset "$D" \
        --task "$task" \
        --n_samples 100 \
        --n_split "$((ncp + 1))" \
        --base_model_name "$M" \
        --output_file "$data_path/${D}_${M}_${task}" \
        --do_temperature
    done
done

trained_model_path="scripts/FineTune/ckpt/"
aux_model="gemma-1b-instruct"

# -------------------------
# main experiment loop
# -------------------------
for M in "${Ms[@]}"; do
  echo "##################################################"
  echo "$(date), Model: $M"
  echo "##################################################"

  for D in "${datasets[@]}"; do
    eval_data_path="$data_path/${D}_${M}_${task}"
    eval_result_path="$res_path/${D}_${M}_${task}"

    echo "=================================================="
    echo "$(date), Dataset: $D (cp_num=$cp_num)"
    echo "=================================================="

    for phi in "${phis[@]}"; do
      echo "---- Phi: $phi ----"

      # phi-specific arguments
      PHI_ARGS=""
      if [ "$phi" = "Bino" ]; then
        PHI_ARGS="--phi Bino --aux_model ${aux_model}"
      elif [ "$phi" = "FDGPT" ]; then
        if [ "$M" != "gpt-4o" ]; then
            PHI_ARGS="--phi FDGPT --base_model ${M} --aux_model ${M}"
        else
            PHI_ARGS="--phi FDGPT --base_model ${aux_model} --aux_model ${aux_model}"
        fi
      else
        PHI_ARGS="--phi FT"
      fi
      
      if [ "$M" != "gpt-4o" ]; then
          BASE_MODEL_ARGS="--base_model ${M}"
      else
          BASE_MODEL_ARGS="--base_model ${aux_model}"
      fi
      # ------------------------------------------------
      # sentence-wise prediction baselines
      # ------------------------------------------------
      python scripts/detector_llm_inquiry.py \
        ${BASE_MODEL_ARGS} \
        --eval_dataset ${eval_data_path} \
        --output_file ${eval_result_path} 

      python scripts/detector_naive_sp.py \
        --from_pretrained ${trained_model_path} \
        ${PHI_ARGS} \
        --eval_dataset ${eval_data_path} \
        --output_file ${eval_result_path}

      python scripts/detector_voting_sp_autothres.py \
        --from_pretrained ${trained_model_path} \
        ${PHI_ARGS} \
        --width 3 \
        --eval_dataset ${eval_data_path} \
        --output_file ${eval_result_path}

      # ------------------------------------------------
      # change-point based methods
      # ------------------------------------------------
      for cp_method in "${cp_methods[@]}"; do
        echo "  >> CP method: $cp_method"

        # vanilla version
        # python scripts/detector_value_cp.py \
        #   --from_pretrained ${trained_model_path} \
        #   ${PHI_ARGS} \
        #   --cp_method ${cp_method} \
        #   --power 0.0 \
        #   --eval_dataset ${eval_data_path} \
        #   --output_file ${eval_result_path}

        # weighted version
        # for power in "${power_list[@]}"; do
        #   python scripts/detector_value_cp.py \
        #     --from_pretrained ${trained_model_path} \
        #     ${PHI_ARGS} \
        #     --cp_method ${cp_method} \
        #     --power ${power} \
        #     --eval_dataset ${eval_data_path} \
        #     --output_file ${eval_result_path}
        # done
        
        # oracle version
        # python scripts/detector_value_cp.py \
        #   --from_pretrained ${trained_model_path} \
        #   ${PHI_ARGS} \
        #   --cp_method 'Oracle' \
        #   --eval_dataset ${eval_data_path} \
        #   --output_file ${eval_result_path}
      done
    done
  done
done
# /usr/bin/shutdown

