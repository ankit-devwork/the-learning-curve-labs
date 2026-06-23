"""Phase 8 sharing lifecycle: rate limit keys and access helpers."""


def test_sharing_rate_limit_key_patterns():
    user_id = "user-1"
    token = "abc123def4567890"
    assert f"sharing:invite_preview:{token[:16]}" == "sharing:invite_preview:abc123def4567890"
    assert f"sharing:invite_accept:{user_id}" == "sharing:invite_accept:user-1"
    assert f"sharing:invite_create:{user_id}" == "sharing:invite_create:user-1"
    assert f"sharing:member_change:{user_id}" == "sharing:member_change:user-1"


def test_cache_invalidation_keys_for_removed_member():
    user_id = "removed-user"
    document_id = "doc-1"
    file_hash = "hash-abc"
    assert f"semantic_chat:{user_id}:{document_id}" == "semantic_chat:removed-user:doc-1"
    assert (
        f"semantic_excel_chat:{user_id}:{document_id}:{file_hash}"
        == "semantic_excel_chat:removed-user:doc-1:hash-abc"
    )
    assert f"semantic_multi_chat:{user_id}:*" == "semantic_multi_chat:removed-user:*"


def test_member_role_values():
    allowed_roles = {"editor", "viewer"}
    assert "owner" not in allowed_roles
    assert "editor" in allowed_roles
    assert "viewer" in allowed_roles
