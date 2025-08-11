def update_counter(counter, stage, condition_met):
    if condition_met and stage == "down":
        stage = "up"
        counter += 1
    elif not condition_met and stage == "up":
        stage = "down"
    return counter, stage
