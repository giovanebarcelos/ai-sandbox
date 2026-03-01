# GO1837-RewardPnl
reward = pnl  # Profit and Loss
       - inventory_risk * 0.1  # Penaliza ter muito estoque
       - order_cancel_fee * 0.01
       + liquidity_rebate  # Exchanges pagam por liquidez
