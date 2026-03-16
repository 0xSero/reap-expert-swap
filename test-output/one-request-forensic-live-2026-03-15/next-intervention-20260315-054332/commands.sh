ssh $REMOTE "pkill -9 -f 'vllm_multiplex_server.py' || true"
scp $PLAN_LOCAL $REMOTE:$PLAN_REMOTE
ssh $REMOTE "export REAP_PLAN_FILE=$PLAN_REMOTE; export CUDA_VISIBLE_DEVICES=2,3; nohup ... vllm_multiplex_server.py ... --gpu-memory-utilization 0.60 ..."
uv run ... evaluate_original_vs_multiplex.py --mode dynamic --sample-count 1 ...
python3 one_request_forensic_replay.py --strict
