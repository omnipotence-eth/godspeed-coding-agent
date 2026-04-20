def list_orders(request):
    return {"orders": request["orders"]}


def cancel_order(request):
    return {"cancelled": request["order_id"]}
