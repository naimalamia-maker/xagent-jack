"""User permissions and access control for RAG tools."""

from typing import Optional


class UserPermissions:
    """Handle user permissions and data access control."""

    @staticmethod
    def get_user_filter(
        user_id: Optional[int], is_admin: bool = False
    ) -> Optional[str]:
        """
        Generate user filter expression for LanceDB queries.

        Args:
            user_id: Current user ID, None for unauthenticated
            is_admin: Whether user is admin

        Returns:
            LanceDB filter expression string, or None for no filtering

        Note:
            Legacy data (user_id = NULL) is only accessible to admin users.
            Regular users can only see data they explicitly own (user_id == their ID).
        """
        if is_admin:
            # Admins can see all data (including NULL user_id legacy data)
            return None
        elif user_id is not None:
            # Regular users can ONLY see their own data
            # Legacy data (NULL user_id) is NOT visible to regular users
            return f"user_id == {user_id}"
        else:
            # Unauthenticated users cannot see any data
            # Return a filter that matches nothing
            return "user_id == -1"  # Impossible condition

    @staticmethod
    def can_access_data(
        user_id: Optional[int], data_user_id: Optional[int], is_admin: bool = False
    ) -> bool:
        """
        Check if user can access specific data.

        Args:
            user_id: Current user ID
            data_user_id: Data owner's user ID
            is_admin: Whether current user is admin

        Returns:
            True if access allowed

        Note:
            Legacy data (data_user_id = NULL) is only accessible to admin users.
            Regular users can only access data they explicitly own.
        """
        if is_admin:
            # Admins can access all data including legacy (NULL) data
            return True
        if user_id is None:
            # Unauthenticated users cannot access any data
            return False
        # Users can ONLY access their own data
        # Legacy data (NULL data_user_id) is NOT accessible to regular users
        return data_user_id == user_id

    @staticmethod
    def get_write_user_id(user_id: Optional[int]) -> Optional[int]:
        """
        Get user_id for writing new data.

        Args:
            user_id: Current user ID

        Returns:
            user_id to use for new data, None for legacy compatibility
        """
        return user_id  # Always use current user_id for new data
