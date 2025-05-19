# # Create a custom wrapper to handle the extra parameter
# class ResearchAssistantWrapper:
#     def __init__(self, qa_chain, memory):
#         self.qa_chain = qa_chain
#         self.memory = memory
#
#     def __call__(self, inputs):
#         # Get chat history from memory
#         chat_history = self.memory.chat_memory.messages
#
#         # Add length preference to inputs if present
#         if "length_preference" in inputs:
#             # Run the chain with the length preference
#             result = self.qa_chain.invoke({
#                 "question": inputs["question"],
#                 "length_preference": inputs["length_preference"],
#                 "chat_history": chat_history
#             })
#         else:
#             # Run the chain without length preference
#             result = self.qa_chain.invoke({
#                 "question": inputs["question"],
#                 "length_preference": "Response Style: Provide a balanced and complete answer with appropriate detail.",
#                 "chat_history": chat_history
#             })
#
#         # Update memory
#         self.memory.chat_memory.add_user_message(inputs["question"])
#         self.memory.chat_memory.add_ai_message(result["answer"])
#
#         # Return result in the expected format
#         return {"answer": result["answer"]}