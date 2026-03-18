## Example usgae 1: python evaltask-v6.py --model meta-llama/Meta-Llama-3-8B-Instruct --tasks math500 aime1k gsm8k aime25 amc olympiad minerva --rollouts 256 --temperature 1.0 --gen_len 7500 --max_sample 50 --output_path l3b8-v-8k --full_logs True --sys_prompt rl_prompt --devices '0,1,2,3,5,6,7,8' --seed 42 --verbose False

## Example usgae 2: python evaltask-v6.py --model /data/home/fyk/saves/l8b-nemo-math-25k-5xlr-e1 --tasks math500 gsm8k amc --rollouts 8 --temperature 0.5 --gen_len 8000 --max_sample 100 --output_path l8b-nemo-math-25k-5xlr-e1-t50-v --full_logs True --sys_prompt rl_prompt --devices '2,3' --seed 42 --verbose True

## Test: python evaltask-v6.py --model /data/home/fyk/saves/l8b-nemo-math-25k-5xlr-e1 --tasks math500 gsm8k amc --rollouts 2 --temperature 0.5 --gen_len 80 --max_sample 10 --output_path l8b-nemo-math-25k-5xlr-e1-t50-test --full_logs True --sys_prompt rl_prompt --devices '0,1' --seed 42 --verbose True


import platform
print(platform.python_version())

import os
os.environ['VLLM_USE_V1'] = '0'
print(f"Disable V1 Engine (epected output: 0):{os.getenv('VLLM_USE_V1')}")
import re, random, pickle, argparse, sys
import numpy as np 
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams
from math_verify import parse, verify, ExprExtractionConfig
from datasets import load_dataset
from tqdm import tqdm
from copy import deepcopy
from scipy.special import comb


import argparse


def reward_correct(item, answer):
    ground_truth = item
    ans = answer
    return 1 if verify(parse(ground_truth), parse(ans)) else 0
    
def reward_format(answer):
    # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else 0

def main():
    parser = argparse.ArgumentParser(description="arguments: --model [str] --tasks [list] --rollouts [int] --temperature [float] --gen_len [int] --max_sample [int] --output_path [str] --sys_prompt [str] --devices [list] --seed [int] --verbose [bool]")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='HuggingFace repo id or local path'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        required=True,
        help='One or more tasks to evaluate, separated by spaces (e.g., aime1k gsm8k). Supported: math500, aime1k, gsm8k, aime25, amc, olymiad, minerva'
    )
    parser.add_argument(
        '--rollouts',
        type=int,
        default=64,
        help='number of rollouts per question (needs to be the power of 2); default: 64'
    )
    parser.add_argument(
        '--gen_len',
        type=int,
        required=True,
        help='Max generation length (<=7500)'
    )    
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='decoding temperature; default: 1.0'
    )
    parser.add_argument(
        '--max_sample',
        type=int,
        default=-1,
        help='Max samples to evaluate for each task; using full test dataset if not provided'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for evaluation results (e.g., llama3-8b-base-aime1k-8k)'
    )
    parser.add_argument(
        '--full_logs',
        type=bool,
        default=False,
        help='Whether to save full generated text to disk. Default: False.'
    )
    parser.add_argument(
        '--sys_prompt',
        type=str,
        default='no_prompt',
        help='System prompt used in evaluations. Default: no system prompt (no_prompt). Alternative: rl_prompt'
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='0,1,2,3,4,5,6,7',
        help='Index for CUDA device to use as a string connected by a comma (e.g., "0,1,2,3")'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for numpy, primarily for subsampling from benchmarks. (Default: 42)'
    )
    parser.add_argument(
        '--verbose',
        type=bool,
        default=False,
        help='Output full generated traces and verification results. (Default: False)'
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        return

    print("--- Evaluation Configuration ---")
    print(f"Model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Rollouts: {args.rollouts}")
    print(f"Generation Length: {args.gen_len}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Samples per Task: {args.max_sample}")
    print(f"Output Path: {args.output_path}")
    print(f"Save Full Logs: {args.full_logs}")
    print(f"System Prompt: {args.sys_prompt}")
    print(f"CUDA Device IDs: {args.devices}")
    print(f"Random Seed: {args.seed}")
    print(f"Verbose: {args.verbose}")
    print("--------------------------")

    model_id = args.model
    task_list = args.tasks
    repn = args.rollouts
    gen_len = args.gen_len
    deco_temp = args.temperature
    m_sample = args.max_sample
    output_path = args.output_path
    full_logs = args.full_logs
    sys_prompt = args.sys_prompt
    cuda_ids = args.devices
    seed = args.seed
    verbose = args.verbose
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_ids


    # Set the seed for reproducibility
    np.random.seed(seed)

    default_params = {
            "gpu_memory_utilization": 0.50,
            'max_model_len': 8192,
            'tensor_parallel_size' : len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        }
    llm = LLM(
            model=model_id,
            **default_params
        )
    
    eval_outputs = []
    crr_outputs = []
    log_outputs = []

    np2 = int(np.log2(repn))
    passks = []
    for i in range(np2+1):
        passks.append(np.power(2,i))
        
    if np.power(2,np2) == repn:
        print("number of rollouts = "+str(repn) + ". evaluating pass@k for k = " + str(passks))
    else:
        repn = int(np.power(2,np2))
        print("the number of rollouts needs to be the power of 2; rounded down to " + str(repn))
        print("number of rollouts = "+str(repn) + ". evaluating pass@k for k = " + str(passks))

    
    file_name = output_path + "_EvalRes.txt"
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            eval_outputs = [file.read()]
        print("\n *** resuming from previous evaluation results. Loaded results: *** \n")
        print(eval_outputs)
        file_name = "Verb_crr_" + output_path + "_EvalCrrs.txt"
        if os.path.exists(file_name):
            with open(file_name, 'r') as file:
                crr_outputs = [file.read()]
        if full_logs:
            file_name = "Verb_trace_" + output_path + "_EvalTexts.txt"
            if os.path.exists(file_name):
                with open(file_name, 'r') as file:
                    log_outputs = [file.read()]
    else:
        print("\n *** no prior output found; running evaluation from start *** \n")
    
    for t in task_list:
        print("\n ***** Running Evaluation on Task: [" + t + "]****** \n")
        if len(eval_outputs)>0:
            if t in eval_outputs[0]:
                print("\n *** evaluation results exist; skip *** \n")
                continue

        if t=="math500":
            dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval",split="test")
            QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['problem'], dataset['solution'])]
            # ==== options for level selection ====
            # ldic = {}
            # for i in tqdm(dataset):
            #     if i['level'] in ldic.keys():
            #         ldic[i['level']].append(i)
            #     else:
            #         ldic[i['level']] = [i]
            # # ldic.keys(),len(ldic['Level 1']),len(ldic['Level 2']),len(ldic['Level 3']),len(ldic['Level 4']),len(ldic['Level 5']), len(ldic['Level ?'])
            # QAs = [{'Q': i['problem'], 'A': i['solution'].split('####')[-1].strip()} for i in ldic['Level 5']]
            if m_sample < len(QAs):
                rndset = np.random.choice(dataset, m_sample, replace=False)
                QAs = [{'Q': i['problem'], 'A': i['solution'].split('####')[-1].strip()} for i in rndset]
        elif t=="aime1k":
            dataset = load_dataset("di-zhang-fdu/AIME_1983_2024",split="train")
            QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['Question'], dataset['Answer'])]
            if m_sample < len(QAs):
                rndset = np.random.choice(dataset, m_sample, replace=False)
                QAs = [{'Q': i['Question'], 'A': i['Answer'].split('####')[-1].strip()} for i in rndset]
        elif t=="gsm8k":
            dataset = load_dataset("openai/gsm8k", "main", split="test")
            QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
            if m_sample < len(QAs):
                rndset = np.random.choice(dataset, m_sample, replace=False)
                QAs = [{'Q': i['question'], 'A': i['answer'].split('####')[-1].strip()} for i in rndset]
        elif t=="aime25":
            dataset = load_dataset("math-ai/aime25",split="test")
            QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['problem'], dataset['answer'])]
            if m_sample < len(QAs):
                rndset = np.random.choice(dataset, m_sample, replace=False)
                QAs = [{'Q': i['problem'], 'A': i['answer'].split('####')[-1].strip()} for i in rndset]
        elif t=="amc":
            dataset = load_dataset("AI-MO/aimo-validation-amc",split="train")
            QAs = [{'Q':x, 'A':str(y).split('####')[-1].strip()} for x,y in zip(dataset['problem'], dataset['answer'])]
            if m_sample < len(QAs):
                rndset = np.random.choice(dataset, m_sample, replace=False)
                QAs = [{'Q': i['problem'], 'A': str(i['answer']).split('####')[-1].strip()} for i in rndset]
        elif t=="olympiad":
            dataset = load_dataset("Hothan/OlympiadBench", "OE_MM_maths_en_COMP", split="train")
            QAs = [{'Q':x, 'A':y[0].split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['final_answer'])]
            if m_sample < len(QAs):
                rndset = np.random.choice(dataset, m_sample, replace=False)
                QAs = [{'Q': i['question'], 'A': i['final_answer'][0].split('####')[-1].strip()} for i in rndset]
        elif t=="minerva":
            dataset = load_dataset("math-ai/minervamath",split="test")
            QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
            if m_sample < len(QAs):
                rndset = np.random.choice(dataset, m_sample, replace=False)
                QAs = [{'Q': i['question'], 'A': i['answer'].split('####')[-1].strip()} for i in rndset]
        else:
            print("Task is not implemented. Available: math500, aime1k, gsm8k, aime25, amc, olymiad, minerva")
            continue

        if sys_prompt == "no_prompt":
            system_prompt = ""
        elif sys_prompt == "rl_prompt":
            system_prompt = "Let's think step by step and output the final answer within \\boxed{}."
        else:
            print("Specified system prompt is not defined. Available: no_prompt, rl_prompt")
            continue
        inputs = QAs
        prompts = [x["Q"] + " " + system_prompt for x in inputs]
    
        tip_text = []
        for x in prompts:
            tip_text.append(x)

        # if full_logs:
        #     log_outputs.append(deepcopy(inputs))
        #     log_outputs.append(deepcopy(tip_text))
        
        # if repn == 1:
        #     print("Greedy Decoding with n=1")
        #     sampling_params = SamplingParams(max_tokens=gen_len, temperature=0.0, logprobs=0)
        # else:
        print("Probablistic Decoding with n=" + str(repn) + " and temperature=" + str(deco_temp)[:4])
        sampling_params = SamplingParams(max_tokens=gen_len, temperature=deco_temp, logprobs=0)


        # ansvec = []
        # logvec = []
        # for rep in range(repn):
        #     print("\n *** Running rollout #"+str(rep)+" of "+str(repn)+" *** \n")
        #     outputs = llm.generate(tip_text, sampling_params)
        #     answers = []
        #     for output in outputs:
        #         prompt = output.prompt
        #         generated_text = output.outputs[0].text
        #         answers.append(generated_text)
        #     # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        #     ansvec.append(deepcopy(answers))
        #     logvec.append(deepcopy(outputs))

        # if full_logs:
        #     log_outputs.append(deepcopy(ansvec))

        ansvec = []
        logvec = []

        tip_textb = []
        for rep in range(repn):
            tip_textb = tip_textb + tip_text

        print("\n *** Running "+str(repn)+" rollout altogether *** \n")
        outputs = llm.generate(tip_textb, sampling_params)
        answers = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            answers.append(generated_text)
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        for rep in range(repn):
            ansvec.append(deepcopy(answers[rep*len(tip_text):(rep+1)*len(tip_text)]))
            logvec.append(deepcopy(outputs[rep*len(tip_text):(rep+1)*len(tip_text)]))

        # if full_logs:
        #     log_outputs.append(deepcopy(ansvec))


        
        # # computing pass@n and avg@n acc
        # crr1 = np.zeros([len(ansvec[0])])
        # crr2 = np.zeros([len(ansvec[0])])
        # for rep in range(len(ansvec)):
        #     answers = ansvec[rep]
        #     for i in tqdm(range(len(answers))):
        #         crr1[i] = np.max([crr1[i], reward_correct(QAs[i]['A'], answers[i])])
        #         crr2[i] += reward_correct(QAs[i]['A'], answers[i])
        # print("Pass@"+str(repn)+" acc="+str(np.mean(crr1)*100)[:5])
        # crr2 = crr2/len(ansvec)
        # print("Avg@"+str(repn)+" acc="+str(np.mean(crr2)*100)[:5])

        
        crr = np.zeros([len(ansvec), len(ansvec[0])])
        allqas = []
        for rep in range(len(ansvec)):
            answers = ansvec[rep]
            for i in tqdm(range(len(answers))):
                crr[rep, i] = reward_correct(QAs[i]['A'], answers[i])
                allqas.append(QAs[i]['Q']+answers[i])

        passres = []
        crrx = np.zeros([len(passks), len(ansvec[0])])
        for i in range(len(passks)):
            ck = passks[::-1][i]
            for j in range(len(ansvec[0])):
                crrx[i,j] = 1 - (1.0*comb(repn-np.sum(crr[:, j]), ck, exact=True))/(1.0*comb(repn, ck, exact=True))
            passres.append(np.average(crrx[i,:]))

        
        # computing entropy and entropy on correct answers
        aent = []
        for rep in range(len(logvec)):
            answers = ansvec[rep]
            for i in tqdm(range(len(answers))):
                # aent.append(logvec[rep][i].outputs[0].cumulative_logprob/len(logvec[rep][i].outputs[0].logprobs))
                aent.append(logvec[rep][i].outputs[0].cumulative_logprob)
        print("Avg Entropy="+str(np.mean(aent))[:6])
        cent = []
        for rep in range(len(logvec)):
            answers = ansvec[rep]
            for i in tqdm(range(len(answers))):
                if crr[rep, i] ==1:
                    # cent.append(logvec[rep][i].outputs[0].cumulative_logprob/len(logvec[rep][i].outputs[0].logprobs))
                    cent.append(logvec[rep][i].outputs[0].cumulative_logprob)
        print("Correct Entropy="+str(np.mean(cent))[:6])

        
        # # Saving Results
        # print("\n *** Saving Results *** \n")
        # eval_outputs.append("Task ["+t+"]: Pass@"+str(repn)+"/Avg@"+str(repn)+"="+str(np.mean(crr1)*100)[:5]+"/"+str(np.mean(crr2)*100)[:5]+"    ")
        # eval_outputs.append("Task ["+t+"]: Avg/Correct Entropy="+str(np.mean(aent))[:6]+"/"+str(np.mean(cent))[:6]+"\n")
        # file_name = output_path + "_EvalRes.txt"
        # try:
        #     print(f"Writing output to '{file_name}'...")
        #     with open(file_name, 'w') as file:
        #         file.writelines(eval_outputs)
        #     print("Successfully wrote to the evalres file.")
        # except IOError as e:
        #     print(f"Error writing to file: {e}")

                # Saving Results
        print("\n *** Saving Results *** \n")

        ostr = "Task ["+t+"]: "
        for i in range(len(passks)):
            ostr = ostr + "Pass@" + str(passks[::-1][i]) + "=" + str(passres[i]*100)[:5] + '/'
            
        eval_outputs.append(ostr+"    ")

        
        eval_outputs.append("Task ["+t+"]: Avg/Correct Entropy="+str(np.mean(aent))[:6]+"/"+str(np.mean(cent))[:6]+"\n")
        file_name = output_path + "_EvalRes.txt"
        try:
            print(f"Writing output to '{file_name}'...")
            with open(file_name, 'w') as file:
                file.writelines(eval_outputs)
            print("Successfully wrote to the evalres file.")
        except IOError as e:
            print(f"Error writing to file: {e}")

        if verbose:
            # Saving Sample Accuracies
            print("\n *** Saving Sample Accuracies *** \n")
            crr_outputs.append("Task ["+t+"]: Avg Acc@"+str(repn)+"\n")
            crr_outputs.append(str(crr.tolist())+"\n")
            file_name = "Verb_crr_" + output_path + "_EvalCrrs.txt"
            try:
                print(f"Writing output to '{file_name}'...")
                with open(file_name, 'w') as file:
                    file.writelines(crr_outputs)
                print("Successfully wrote to the evalcrrs file.")
            except IOError as e:
                print(f"Error writing to file: {e}")
    
    
            # Saving Full Traces
            if full_logs:
                print("\n *** Saving Full Traces *** \n")
                log_outputs.append(str(allqas) + '\n')
                # log_outputs.append("Task ["+t+"]: Pass@"+str(repn)+"/Avg@"+str(repn)+"="+str(np.mean(crr1)*100)[:5]+"/"+str(np.mean(crr2)*100)[:5]+"\n")
                file_name = "Verb_trace_" + output_path + "_EvalTexts.txt"
                try:
                    print(f"Writing output to '{file_name}'...")
                    with open(file_name, 'w') as file:
                        file.writelines(str(log_outputs))
                    print("Successfully wrote to the evaltexts file.")
                except IOError as e:
                    print(f"Error writing to file: {e}")
    
    print("All Evaluations Finished Successfully.")
        
if __name__ == "__main__":
    main()

