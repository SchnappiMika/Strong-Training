from .envs import timberman_action_hazard


def check_focused_timber_policy(agent, replay):
    observed = {}
    for row in replay.online.rows + replay.history["timberman"].rows:
        if row["task"] != "timberman":
            continue
        state = row["state"]
        if state[0] == state[-1]:
            continue
        key = (state[0], state[1], state[-1])
        observed.setdefault(key, state)
    failures = []
    for key, state in observed.items():
        action, embedding = agent.select(state, "timberman", deterministic=True)
        hazard = timberman_action_hazard(state, action)
        if hazard:
            failures.append({"observed_pattern": list(key), "action": action, "hazard": hazard})
    return {"checked_observed_patterns": len(observed), "failures": failures,
            "passed": bool(observed) and not failures,
            "scope": "Regression diagnostic on replay observations; does not alter actions or establish game score"}
