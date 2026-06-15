from __future__ import annotations

import streamlit as st
from PIL import Image

from config import settings
from src.agent.agent import PlantCareAgent
from src.agent.tools import SessionContext
from src.utils.logging_config import setup_logging

setup_logging()

st.set_page_config(page_title="PlantCare AI", page_icon="🌿", layout="centered")


def init_state() -> None:
    if "context" not in st.session_state:
        st.session_state.context = SessionContext()
    if "agent" not in st.session_state:
        st.session_state.agent = PlantCareAgent(st.session_state.context)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Cześć! 🌿 Jestem **PlantCare AI**. Prześlij zdjęcie swojej rośliny "
                    "domowej, a rozpoznam jej gatunek i podpowiem, jak o nią dbać. "
                    "Możesz też po prostu zapytać o konkretną roślinę."
                ),
                "steps": [],
            }
        ]


def reset_conversation() -> None:
    for key in ("context", "agent", "messages"):
        st.session_state.pop(key, None)
    init_state()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("🌿 PlantCare AI")
        st.caption("Rozpoznawanie i pielęgnacja roślin domowych")

        missing = settings.validate()
        if missing:
            st.error("Brak kluczy API: " + ", ".join(missing) + ". Uzupełnij plik `.env`.")
        else:
            st.success("Konfiguracja API OK")

        st.divider()
        uploaded = st.file_uploader(
            "Prześlij zdjęcie rośliny", type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded is not None:
            image = Image.open(uploaded)
            st.session_state.context.image = image
            st.image(image, caption="Twoja roślina", use_container_width=True)
            st.info("Zdjęcie gotowe. Napisz np. „Co to za roślina i jak o nią dbać?")

        ctx = st.session_state.context
        if ctx.identified_common:
            st.divider()
            st.metric("Rozpoznano", ctx.identified_common,
                      f"pewność {ctx.identified_confidence:.0%}")
            if ctx.rag.sources:
                with st.expander(f"Źródła ({len(ctx.rag.sources)})"):
                    for s in ctx.rag.sources:
                        st.write(f"- {s}")

        st.divider()
        st.button("🗑️ Nowa rozmowa", on_click=reset_conversation, use_container_width=True)


def render_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else "🧑"):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("🔎 Kroki agenta"):
                    for tool, obs in msg["steps"]:
                        st.markdown(f"**{tool}** → {obs}")


def handle_user_message(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt, "steps": []})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Myślę..."):
            try:
                result = st.session_state.agent.chat(prompt)
                answer, steps = result["output"], result["steps"]
            except Exception as exc:
                answer, steps = f"Wystąpił błąd: {exc}", []
        st.markdown(answer)
        if steps:
            with st.expander("🔎 Kroki agenta"):
                for tool, obs in steps:
                    st.markdown(f"**{tool}** → {obs}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "steps": steps})


def main() -> None:
    init_state()
    render_sidebar()
    st.title("🌿 PlantCare AI")
    st.caption("Twój asystent do rozpoznawania i pielęgnacji roślin domowych")
    render_chat()

    if prompt := st.chat_input("Zapytaj o swoją roślinę..."):
        handle_user_message(prompt)


if __name__ == "__main__":
    main()
