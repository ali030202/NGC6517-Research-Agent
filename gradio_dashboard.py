#!/usr/bin/env python3
"""Compatibility launcher for the unified research dashboard."""
from gradio_app import create_app


if __name__ == "__main__":
    create_app().queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
