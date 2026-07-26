"""
fl_runner.py
============
Main federated learning runner.
Wires FederatedServer + FLClient instances together
into a complete FL training loop.

This is the correct architectural implementation:
  - Server and clients are separate objects
  - Clients hold private data (never shared)
  - Only gradient updates cross the server-client boundary
  - Three-layer defence runs entirely on the server
  - Async mode supported (clients drop out with probability p_churn)

Usage (replaces the monolithic loop in experiment1_baselines.py):

  from fl_runner import build_federation, run_federation

  server, clients = build_federation(X, y, scaler, X_val, y_val,
                                     n_clients=100,
                                     byzantine_ratio=0.20,
                                     attack_type='lfp')
  history = run_federation(server, clients, n_rounds=50)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from fl_model   import FLNeuralNet
from fl_client  import FLClient
from fl_server  import FederatedServer
from fl_base    import partition_noniid, make_validation_set
from aggregators import MultiLayerAggregator


# ── Build complete federation (server + all clients) ──────────────────────────
def build_federation(X, y, scaler, X_val, y_val,
                     n_clients=100,
                     clients_per_round=20,
                     byzantine_ratio=0.20,
                     attack_type='lfp',
                     noniid_alpha=0.5,
                     lr=0.01,
                     local_epochs=5,
                     batch_size=32,
                     async_mode=False,
                     max_staleness=5,
                     staleness_corrected_kappa=True,
                     tau=2.0, alpha_trust=0.8, delta=0.2,
                     seed=42):
    """
    Instantiate a complete federated system.

    Parameters
    ----------
    X, y          : full dataset (will be partitioned across clients)
    scaler        : fitted StandardScaler (used by server for Kappa eval)
    X_val, y_val  : server-held validation set D_val
    n_clients     : total clients in the federation
    byzantine_ratio: fraction of clients that are adversarial
    attack_type   : 'lfp' | 'grad-p' | 'spf' | 'ooa'
    noniid_alpha  : Dirichlet alpha for data partitioning
    async_mode    : True = asynchronous FL with client churn
    tau/alpha/delta: three-layer defence hyperparameters

    Returns
    -------
    server  : FederatedServer
    clients : list of FLClient
    """
    rng = np.random.RandomState(seed)
    n_features = X.shape[1]

    # ── Partition data across clients (Non-IID) ───────────────────────
    client_data = partition_noniid(
        X, y, n_clients, alpha=noniid_alpha, seed=seed)

    # ── Identify Byzantine clients ────────────────────────────────────
    n_byzantine  = int(n_clients * byzantine_ratio)
    n_benign     = n_clients - n_byzantine
    byzantine_ids= set(rng.choice(n_clients, n_byzantine, replace=False))

    # ── Instantiate clients ───────────────────────────────────────────
    clients = []
    for cid in range(n_clients):
        Xc, yc    = client_data[cid]
        is_byz    = cid in byzantine_ids
        client    = FLClient(
            client_id    = cid,
            X_local      = Xc,
            y_local      = yc,
            n_features   = n_features,
            is_byzantine = is_byz,
            attack_type  = attack_type if is_byz else None,
            lr           = lr,
            local_epochs = local_epochs,
            batch_size   = batch_size,
        )
        clients.append(client)

    # ── Instantiate aggregator ────────────────────────────────────────
    aggregator = MultiLayerAggregator(
        n_clients   = n_clients,
        tau         = tau,
        alpha       = alpha_trust,
        delta       = delta,
    )

    # ── Instantiate server ────────────────────────────────────────────
    server = FederatedServer(
        n_features       = n_features,
        n_clients        = n_clients,
        clients_per_round= clients_per_round,
        aggregator       = aggregator,
        X_val            = X_val,
        y_val            = y_val,
        scaler           = scaler,
        lr               = lr,
        async_mode       = async_mode,
        max_staleness    = max_staleness,
        staleness_corrected_kappa = staleness_corrected_kappa,
    )

    n_byz_str = f"{n_byzantine} Byzantine [{attack_type}]"
    n_ben_str = f"{n_benign} benign"
    print(f"  Federation: {n_clients} clients "
          f"({n_byz_str}, {n_ben_str})")
    print(f"  Model     : {FLNeuralNet(n_features)}")
    print(f"  Mode      : {'Async' if async_mode else 'Sync'} FL")

    return server, clients, list(byzantine_ids)


# ── Run complete FL training ───────────────────────────────────────────────────
def run_federation(server, clients, byzantine_ids,
                   n_rounds=50,
                   p_churn=0.0,
                   verbose=True,
                   seed=42):
    """
    Run the full federated training loop.

    Parameters
    ----------
    server        : FederatedServer
    clients       : list of FLClient
    byzantine_ids : set of Byzantine client IDs
    n_rounds      : number of FL rounds
    p_churn       : probability a client drops out each round (0 = no churn)
    verbose       : print round-level metrics

    Returns
    -------
    history : list of per-round metric dicts
    """
    history = []

    for rnd in range(n_rounds):
        rng = np.random.RandomState(seed + rnd)

        # ── Select clients for this round ─────────────────────────────
        n_per_round = server.clients_per_round
        n_byz_round = max(1, int(n_per_round *
                          len(byzantine_ids) / len(clients)))
        n_ben_round = n_per_round - n_byz_round

        byz_pool = [c for c in clients if c.client_id in byzantine_ids]
        ben_pool = [c for c in clients if c.client_id not in byzantine_ids]

        sampled_ben = list(rng.choice(len(ben_pool),
                           min(n_ben_round, len(ben_pool)),
                           replace=False))
        sampled_byz = list(rng.choice(len(byz_pool),
                           min(n_byz_round, len(byz_pool)),
                           replace=False))

        selected = ([ben_pool[i] for i in sampled_ben] +
                    [byz_pool[i] for i in sampled_byz])

        # ── Apply churn (some clients drop out) ───────────────────────
        if p_churn > 0:
            selected = [c for c in selected
                        if c.is_available(1 - p_churn, rng)]
            if not selected:
                continue

        # ── Server broadcasts current global model ────────────────────
        server.broadcast(selected)

        # ── Each client trains locally and submits update ─────────────
        # This is the FL boundary: only update vectors cross here
        for client in selected:
            update = client.local_train_and_submit()
            if update is not None:
                server.receive_update(
                    client.client_id,
                    update,
                    sent_at_round=rnd
                )

        # ── Server runs defence + aggregation ─────────────────────────
        metrics = server.aggregate()
        if metrics is None:
            continue

        history.append(metrics)

        if verbose and (rnd % 10 == 0 or rnd == n_rounds - 1):
            n_flagged = metrics['n_flagged']
            n_total   = metrics['n_updates']
            print(f"  Round {rnd:3d} | "
                  f"Acc={metrics['accuracy']:5.1f}% | "
                  f"F1={metrics['f1']:5.1f}% | "
                  f"Flagged={n_flagged}/{n_total} | "
                  f"κ̄={metrics['mean_kappa']:.3f} | "
                  f"Agg={metrics['agg_time_ms']:.1f}ms")

    return history


# ── Run buffered semi-async FL (FedBuff) ──────────────────────────────────────
def run_federation_async(server, clients, byzantine_ids,
                         n_rounds=50,
                         concurrency=None,
                         arrivals_per_tick=None,
                         max_delay=3,
                         p_churn=0.0,
                         verbose=True,
                         seed=42):
    """
    Buffered semi-asynchronous FL (FedBuff-style).

    Unlike run_federation (which trains one synchronous cohort per round),
    here clients train on whatever global *version* they last received and
    their updates arrive after a random delay. The server buffers arrivals
    and performs an aggregation ("model version bump") once `concurrency`
    updates have accumulated, down-weighting each update by its staleness
    (server-model versions elapsed since it trained) via the aggregator's
    staleness term. This models the trusted-then-defector / lagging-worker
    dynamics of spatial-crowdsourcing far better than sync FL.

    Requires server.async_mode = True (staleness weighting + staleness-based
    rejection beyond server.max_staleness are only active in async mode).

    Parameters
    ----------
    concurrency       : updates to buffer before an aggregation
                        (default: server.clients_per_round)
    arrivals_per_tick : new clients dispatched each simulation tick
                        (default: concurrency)
    max_delay         : max arrival delay in ticks (staleness spread)
    n_rounds          : number of aggregations (global-model versions) to run

    Returns
    -------
    history : list of per-round metric dicts (same schema as run_federation,
              each also carrying mean/max staleness of the aggregated batch)
    """
    if not server.async_mode:
        raise ValueError(
            "run_federation_async requires server.async_mode=True "
            "(build_federation(..., async_mode=True)).")

    K = concurrency or server.clients_per_round
    arrivals_per_tick = arrivals_per_tick or K

    byz_pool = [c for c in clients if c.client_id in byzantine_ids]
    ben_pool = [c for c in clients if c.client_id not in byzantine_ids]
    byz_frac = len(byz_pool) / max(len(clients), 1)

    history   = []
    in_flight = []          # updates computed but not yet 'arrived'
    tick      = 0
    # generous tick budget so we can always reach n_rounds aggregations
    max_ticks = n_rounds * (max_delay + 2) + 100

    while len(history) < n_rounds and tick < max_ticks:
        rng = np.random.RandomState(seed + tick)

        # ── 1. Dispatch new participants on the CURRENT global version ────
        n_byz = max(0, int(round(arrivals_per_tick * byz_frac)))
        n_ben = arrivals_per_tick - n_byz
        picks = []
        if ben_pool and n_ben:
            picks += [ben_pool[i] for i in
                      rng.choice(len(ben_pool), min(n_ben, len(ben_pool)),
                                 replace=False)]
        if byz_pool and n_byz:
            picks += [byz_pool[i] for i in
                      rng.choice(len(byz_pool), min(n_byz, len(byz_pool)),
                                 replace=False)]

        server.broadcast(picks)                       # tags round_number=version
        for c in picks:
            if p_churn > 0 and not c.is_available(1 - p_churn, rng):
                continue
            update = c.local_train_and_submit()
            if update is None:
                continue
            delay = int(rng.randint(0, max_delay + 1))
            in_flight.append({
                'client_id':       c.client_id,
                'update':          update,
                'sent_at_version': server.round,       # global version trained on
                'arrival_tick':    tick + delay,
            })

        # ── 2. Deliver everything that has 'arrived' into the buffer ─────
        still = []
        for item in in_flight:
            if item['arrival_tick'] <= tick:
                # receive_update computes staleness = server.round - sent_at
                server.receive_update(item['client_id'], item['update'],
                                      sent_at_round=item['sent_at_version'])
            else:
                still.append(item)
        in_flight = still

        # ── 3. Aggregate once enough updates have buffered ───────────────
        if len(server._async_buffer) >= K:
            stale_batch = [b['staleness'] for b in server._async_buffer]
            metrics = server.aggregate()              # bumps server.round
            if metrics is not None:
                metrics['mean_staleness'] = round(float(np.mean(stale_batch)), 2)
                metrics['max_staleness']  = int(np.max(stale_batch))
                history.append(metrics)
                if verbose and (len(history) % 10 == 0 or len(history) == n_rounds):
                    print(f"  Agg {len(history):3d} | "
                          f"Acc={metrics['accuracy']:5.1f}% | "
                          f"F1={metrics['f1']:5.1f}% | "
                          f"Flagged={metrics['n_flagged']}/{metrics['n_updates']} | "
                          f"stale μ={metrics['mean_staleness']} "
                          f"max={metrics['max_staleness']}")
        tick += 1

    if len(history) < n_rounds and verbose:
        print(f"  [async] stopped early: {len(history)}/{n_rounds} aggregations "
              f"after {tick} ticks (raise arrivals_per_tick or lower concurrency)")
    return history


# ── Quick integration test ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Integration test: full federated training loop")
    print("="*55)

    from dataset_config import get_dataset, DATASET_NAME
    from fl_base import make_validation_set
    from sklearn.preprocessing import StandardScaler
    import sys
    sys.argv = ['test', '--dataset', 'nyc-taxi']

    print(f"\nLoading dataset...")
    X, y = get_dataset(seed=42)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_val, y_val = make_validation_set(X_scaled, y, n=1000, seed=42)

    print(f"\nBuilding federation...")
    server, clients, byz_ids = build_federation(
        X_scaled, y, scaler, X_val, y_val,
        n_clients        = 50,
        clients_per_round= 10,
        byzantine_ratio  = 0.20,
        attack_type      = 'lfp',
        noniid_alpha     = 0.5,
        async_mode       = False,
        seed             = 42,
    )

    print(f"\nRunning 10 FL rounds...")
    history = run_federation(
        server, clients, set(byz_ids),
        n_rounds = 10,
        p_churn  = 0.0,
        verbose  = True,
        seed     = 42,
    )

    final = history[-1]
    print(f"\nFinal accuracy : {final['accuracy']}%")
    print(f"Final F1       : {final['f1']}%")
    print(f"Mean Kappa     : {final['mean_kappa']}")
    print(f"\nIntegration test passed ✓")
