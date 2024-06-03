import streamlit as st
from prisma import Prisma
import asyncio

# Initialize Prisma client
prisma = Prisma()

async def main():
    await prisma.connect()

    # Add new todo item
    with st.form("add_todo"):
        new_todo_title = st.text_input("New Todo Title")
        submitted = st.form_submit_button("Add Todo")
        if submitted and new_todo_title:
            await prisma.todoitem.create({
                'title': new_todo_title,
                'done': False,
            })
            st.success("Todo added!")

    # Display todo items
    todos = await prisma.todoitem.find_many(order={'createdAt': 'desc'})
    for todo in todos:
        col1, col2 = st.columns([2,1])
        with col1:
            checked = st.checkbox(f"{todo.title}", value=todo.done, key=f"done_{todo.id}")
            if checked != todo.done:
                await prisma.todoitem.update(
                    where={'id': todo.id},
                    data={'done': checked},
                )
                st.rerun()
        with col2:
            if st.button("Delete", key=f"delete_{todo.id}"):
                await prisma.todoitem.delete(where={'id': todo.id})
                st.rerun()

    await prisma.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
    
    
# If you are facing the issue of adjustText not found, run the app with the full command
# `python3 -m streamlit run app/12_streamlit_todo.py`
