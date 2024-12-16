# TODO: unify three tasks

def get_context(node, args, backend):
    if "gpt-" in backend:
        messages = get_messages_from_bottom(node)
        context =  messages
    elif "Phi-3" in backend or "llama31" in backend:
        conv = get_conv_from_bottom(node, args.conv_template)
        conv.append_message(conv.roles[1], None)
        context =  conv
    else:
        raise NotImplementedError
    return context