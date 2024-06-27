import bpy

def console_get():
    for area in bpy.context.screen.areas:
        if area.type == 'CONSOLE':
            for space in area.spaces:
                if space.type == 'CONSOLE':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            return area, space, region
    return None, None, None


def print(*args):
    area, space, region = console_get()
    if space is None:
        return

    context_override = bpy.context.copy()
    context_override.update({
        "space": space,
        "area": area,
        "region": region,
    })
    with bpy.context.temp_override(**context_override):
        text = ' '.join([str(e) for e in args])
        for line in text.split("\n"):
            bpy.ops.console.scrollback_append(text=line, type='OUTPUT')
