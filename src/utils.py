from datetime import datetime


def build_markdown_export(messages: list[dict]) -> str:
    lines = [
        "# ResearchPilot conversation",
        f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]
    for message in messages:
        role = "You" if message["role"] == "user" else "ResearchPilot"
        lines.extend([f"## {role}", "", message["content"], ""])
        for source in message.get("sources", []):
            lines.append(
                f"- **{source['document']} · Page {source['page']}** — {source['excerpt']}"
            )
        lines.append("")
    return "\n".join(lines)
