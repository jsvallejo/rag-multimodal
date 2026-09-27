"""
Cliente Streamlit para el sistema RAG Multimodal.
Cumple: chat interactivo con historial, markdown, imagen relacionada junto
a la respuesta, y metadata de fuente (archivo + página).
"""
import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Multimodal", page_icon="📄", layout="wide")
st.title("📄 Asistente de Documentos Técnicos (RAG Multimodal)")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Subir documento")
    uploaded_file = st.file_uploader("PDF técnico", type=["pdf"])

    if uploaded_file and st.button("Procesar documento"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/documents/upload", files=files)

        if response.status_code == 202:
            job_id = response.json()["job_id"]
            st.session_state["job_id"] = job_id
            st.success(f"Documento en procesamiento. Job ID: {job_id}")
        else:
            st.error(f"Error al subir el documento: {response.text}")

    if "job_id" in st.session_state:
        if st.button("Consultar estado de ingesta"):
            status_resp = requests.get(f"{API_URL}/documents/jobs/{st.session_state['job_id']}")
            if status_resp.status_code == 200:
                data = status_resp.json()
                st.info(f"Estado: **{data['status']}**")
                if data["status"] == "failed":
                    st.error(data.get("error_message", "Error desconocido"))

# --- Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for img_url in message.get("images", []):
            st.image(f"{PUBLIC_API_URL}{img_url}", width=400)
        if message.get("sources"):
            with st.expander("📚 Fuentes"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source_file']}**, Página {source['page_number']} "
                        f"(relevancia: {source['score']:.2f})"
                    )
                    st.caption(source["text_snippet"])

question = st.chat_input("Preguntá algo sobre el documento...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el documento..."):
            response = requests.post(f"{API_URL}/query", json={"question": question})

        if response.status_code == 200:
            data = response.json()
            st.markdown(data["answer"])

            if data["insufficient_context"]:
                st.warning("⚠️ El sistema no encontró suficiente contexto para responder con certeza.")

            image_urls = []
            for source in data["sources"]:
                image_urls.extend(source.get("image_urls", []))
                for img_url in source.get("image_urls", []):
                    st.image(f"{API_URL}{img_url}", width=400)

            if data["sources"]:
                with st.expander("📚 Fuentes"):
                    for source in data["sources"]:
                        st.markdown(
                            f"**{source['source_file']}**, Página {source['page_number']} "
                            f"(relevancia: {source['score']:.2f})"
                        )
                        st.caption(source["text_snippet"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "images": image_urls,
                "sources": data["sources"],
            })
        else:
            st.error(f"Error al consultar: {response.text}")
