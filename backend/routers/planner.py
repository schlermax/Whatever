from flask import Blueprint, jsonify, request

from app import generate_lesson_plan, _vector_store, _embeddings_model

planner_bp = Blueprint('planner', __name__, url_prefix='/api')


@planner_bp.route('/planner', methods=['POST'])
def planner():
    """Planner endpoint that generates lesson plans using RAG pipeline.
    
    Request JSON:
        {
            "query": "Teach me about data structures",
            "k": 3,
            "system_instruction": "Optional custom instruction"
        }
    
    Response JSON:
        {
            "success": true,
            "lesson_plan": "Generated lesson plan text",
            "query": "Original user query"
        }
    """
    if not _vector_store or not _embeddings_model:
        return jsonify({"success": False, "error": "RAG pipeline not initialized"}), 500
    
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"success": False, "error": "Missing 'query' field in request"}), 400
    
    user_query = data.get("query")
    k = data.get("k", 3)
    system_instruction = data.get("system_instruction")
    
    try:
        lesson_plan = generate_lesson_plan(
            user_query=user_query,
            vector_store=_vector_store,
            embeddings_model=_embeddings_model,
            k=k,
            system_instruction=system_instruction,
        )
        return jsonify({
            "success": True,
            "query": user_query,
            "lesson_plan": lesson_plan,
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500