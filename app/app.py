import base64

import streamlit as st
from app_utils import inference


def main() -> None:
    st.title("SED Visualization App")

    idx = st.number_input("Enter the index:", value=97)

    try:
        vis, audio, fs = inference(
            idx, model_path="../dvclive/artifacts/critical-bond.ckpt", model="teacher"
        )
    except IndexError:
        st.error("Invalid index. Please enter a valid index.")
        return

    vis.save(filename="vis.png")

    with open("vis.png", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{b64_string}"/></div>',
        unsafe_allow_html=True,
    )

    st.audio(audio, format="audio/wav", start_time=0, sample_rate=fs)


if __name__ == "__main__":
    main()
