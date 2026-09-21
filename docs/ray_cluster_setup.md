# Ray Cluster Setup (Phase 5+ prep)

Not yet tested end-to-end -- no second machine available yet. These are
the commands to run once the home machine is ready, based on Ray's
standard multi-node cluster setup.

## On the head node (likely the more powerful home machine)

1. Confirm both machines can reach each other on the network (same
   LAN, or VPN if remote). Note the head node's IP address.

2. Start the head node:
   ray start --head --port=6379

   This will print the exact command to run on worker nodes, including
   the head node's address -- copy it rather than retyping by hand.

## On each worker node (e.g. this laptop)

3. Start the worker, pointing at the head node's address:
   ray start --address='<head_node_ip>:6379'

## Verifying the cluster

4. From either machine, run:
   ray status

   This should show both nodes connected, with their available
   CPU/GPU resources listed.

## Running training against the cluster

5. In the training script, PPOConfig's .env_runners(num_env_runners=N)
   can now be set higher than the CPU count of a single machine, since
   Ray will distribute EnvRunner actors across all connected nodes
   automatically -- no code changes needed beyond that number.

## Shutting down

6. On each worker: ray stop
   Then on the head node: ray stop

## Open questions for when this is actually set up
- Network setup: same LAN vs. VPN/Tailscale if machines aren't
  physically together
- Whether the home machine has a GPU Ray should be told to use
  (num_gpus in the config)
- Firewall/port considerations for port 6379 and Ray's other ports