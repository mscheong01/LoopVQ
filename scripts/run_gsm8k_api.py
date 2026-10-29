# Drives lm_eval (MIT) over HTTP.
"""Standard (non-reasoning) lm_eval driver against a running SGLang server,
for validating a new SGLang-served quantizer against this project's own
established HF-harness numbers (run_lm_eval.py: apply_chat_template=False, no
gen_kwargs override -- plain few-shot completion, task's own default greedy
decoding). Mirrors run_reasoning_api.py's server-driver structure.
"""
import argparse
import json
import re

import lm_eval
import lm_eval.tasks
from lm_eval.utils import make_table

# lm_eval's mbpp `extract_code_blocks` reads a completion that starts with "def ..." as a fenced
# block tagged "def" and drops the word; require a newline after the language tag.
def _extract_code_blocks_fixed(text: str) -> str:
    pattern = r"```(?:\w+\n)?(.*?)\n?```"
    matches = re.findall(pattern, "```" + text, re.DOTALL)
    if not matches:
        text_without_lang = re.sub(r"```python", "```", text)
        matches = re.findall(pattern, text_without_lang, re.DOTALL)
    return matches[0] if matches else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1/completions")
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_concurrent", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./gsm8k_output")
    parser.add_argument("--save_postfix", type=str, default="run")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--chat", action="store_true",
                         help="use local-chat-completions + apply_chat_template=True "
                              "instead of the default plain-completion protocol. NOTE: "
                              "SGLang's /v1/chat/completions does not honor a trailing "
                              "assistant message as a continuation (verified directly -- "
                              "it starts a fresh turn instead), so this breaks any task "
                              "using gen_prefix (e.g. humaneval_instruct, mbpp_instruct). "
                              "Use --chat_via_completions for those instead.")
    parser.add_argument("--chat_via_completions", action="store_true",
                         help="apply_chat_template=True but render it locally via the HF "
                              "tokenizer (tokenized_requests=True) and send the result to "
                              "the plain /v1/completions endpoint -- correctly honors "
                              "continue_final_message/gen_prefix since the continuation is "
                              "baked into the raw prompt tokens rather than relying on the "
                              "server to understand 'continue this assistant message'.")
    parser.add_argument("--fewshot_as_multiturn", action="store_true",
                         help="with --chat/--chat_via_completions: render few-shot exemplars "
                              "as separate user/assistant turns instead of concatenating them "
                              "into one turn (matches run_lm_eval.py's chat-template protocol)")
    parser.add_argument("--include_path", type=str, default=None,
                         help="extra directory of custom lm_eval task yamls to register "
                              "(e.g. the math500 custom task)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="lm_eval API-model max_length (context+generation budget). The "
                             "TemplateAPI default is tiny (2048) and it truncates the prompt "
                             "to max_length - max_gen_toks -- with a long-generation task like "
                             "aime24 (max_gen_toks=32768) that slice goes NEGATIVE and the "
                             "prompt is silently truncated to EMPTY ('Prompt cannot be empty' "
                             "400s). Set to the serving context length.")
    parser.add_argument("--gen_kwargs", type=str, default=None,
                        help="lm_eval gen_kwargs override string, e.g. "
                             "'temperature=0.6,top_p=0.95,do_sample=True,max_gen_toks=30000' — "
                             "needed for sampled thinking-mode runs.")
    parser.add_argument("--think", action="store_true",
                         help="pass enable_thinking=True into apply_chat_template. Ouro's "
                              "-Thinking variants gate their "
                              "reasoning block on this kwarg and DEFAULT IT OFF, so without "
                              "it they answer like a weak instruct model (Ouro-1.4B-Thinking "
                              "scored 0.58 on GSM8K vs the base model's 0.76 that way).")
    args = parser.parse_args()
    assert not (args.chat and args.chat_via_completions), \
        "--chat and --chat_via_completions are mutually exclusive"

    if args.think:
        from typing import Dict, List
        from lm_eval.models.api_models import JsonChatStr, TemplateAPI

        def _apply_chat_template_think(self, chat_history: List[Dict[str, str]],
                                        add_generation_prompt: bool = True):
            if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
                return self.tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=not add_generation_prompt,
                    enable_thinking=True,
                )
            elif self.tokenizer_backend == "remote" and self.tokenized_requests:
                return chat_history
            else:
                return JsonChatStr(json.dumps(chat_history, ensure_ascii=False))

        TemplateAPI.apply_chat_template = _apply_chat_template_think

    task_manager = lm_eval.tasks.TaskManager(include_path=args.include_path)

    if "mbpp" in args.task:
        import sys
        # Let lm_eval's own loader import the task module first (get_task_dict), then patch the
        # cached module; a module patched before that is replaced by a fresh unpatched copy.
        lm_eval.tasks.get_task_dict([args.task], task_manager)
        mod_name = "lm_eval.tasks.mbpp.utils"
        if mod_name in sys.modules:
            sys.modules[mod_name].extract_code_blocks = _extract_code_blocks_fixed



    model_type = "local-chat-completions" if args.chat else "local-completions"
    base_url = args.base_url
    if args.chat and base_url.rstrip("/").endswith("/completions") and "chat" not in base_url:
        base_url = base_url.rstrip("/").rsplit("/", 1)[0] + "/chat/completions"

    apply_chat_template = args.chat or args.chat_via_completions

    results = lm_eval.simple_evaluate(
        model=model_type,
        model_args={
            "model": args.model_name,
            "base_url": base_url,
            "num_concurrent": args.num_concurrent,
            "max_retries": 3,
            "tokenized_requests": args.chat_via_completions,
            "timeout": args.timeout,
            **({"max_length": args.max_length} if args.max_length else {}),
        },
        tasks=[args.task],
        task_manager=task_manager,
        log_samples=True,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        confirm_run_unsafe_code=True,
        gen_kwargs=args.gen_kwargs,
    )

    print(make_table(results))

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.save_postfix}_{args.seed}_{args.task}.json")
    with open(output_file, "w") as f:
        json.dump(results["results"], f, indent=4)
    print(f"Results saved to {output_file}")

    samples = results.get("samples")
    if samples:
        samples_file = output_file[:-5] + "_samples.jsonl"
        with open(samples_file, "w") as f:
            for task, recs in samples.items():
                for r in recs:
                    # keep the raw generation (`resps`), not only the extracted answer
                    row = {"task": task, "doc_id": r.get("doc_id"), "filter": r.get("filter"),
                           "target": r.get("target"), "filtered_resps": r.get("filtered_resps"),
                           "resps": r.get("resps")}
                    row.update({k: v for k, v in r.items()
                                if k in ("exact_match", "em", "f1", "acc", "finish_reason")})
                    f.write(json.dumps(row, default=str) + "\n")
        print(f"Per-example samples saved to {samples_file}")


if __name__ == "__main__":
    main()
