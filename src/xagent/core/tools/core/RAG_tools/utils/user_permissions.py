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
            Legacy data (user_id = NULL) is accessible to all authenticated users.
            Regular users can see their own data OR legacy data.
        """
        if is_admin:
            # Admins can see all data (including NULL user_id legacy data)
            return None
        elif user_id is not None:
            # Regular users can see their own data OR legacy data (NULL user_id)
            return f"user_id == {user_id} or user_id is null"
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
            Legacy data (data_user_id = NULL) is accessible to all authenticated users.
            Regular users can access their own data OR legacy data.
        """
        if is_admin:
            # Admins can access all data including legacy (NULL) data
            return True
        if user_id is None:
            # Unauthenticated users cannot access any data
            return False
        # Users can access their own data OR legacy data (NULL data_user_id)
        return data_user_id == user_id or data_user_id is None

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
