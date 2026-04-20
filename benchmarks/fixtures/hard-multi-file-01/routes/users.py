def get_user(request):
    return {"id": request["user_id"]}


def create_user(request):
    return {"ok": True, "name": request["name"]}
