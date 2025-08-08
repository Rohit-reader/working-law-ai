def truncate_text(text, max_lines=4):
    """Truncate text to specified number of lines"""
    if not text:
        return ""
    lines = text.split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + '...'
    return text
