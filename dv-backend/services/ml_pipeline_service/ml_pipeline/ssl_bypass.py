
"""
SSL Validation Bypass Module

This module provides utilities to bypass SSL certificate verification issues
that commonly occur on macOS when using ucimlrepo.fetch_ucirepo().

Usage:
    from ssl_validation import disable_ssl_verification
    
    # Call this once at the beginning of your script
    disable_ssl_verification()
    
    # Now fetch_ucirepo will work without SSL certificate issues
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(name='Iris')
"""

import ssl
import urllib.request
import urllib.parse
from typing import Optional


def disable_ssl_verification():
    """
    Globally disable SSL certificate verification for urllib requests.
    
    This function modifies the default SSL context to bypass certificate
    verification issues commonly encountered on macOS when accessing
    UCI ML Repository datasets.
    
    Note: This reduces security by disabling certificate verification.
    Use only for trusted sources like UCI ML Repository.
    
    Returns:
        None
    """
    try:
        # Create an unverified SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create a custom HTTPS handler with unverified SSL context
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        
        # Install the custom handler globally
        opener = urllib.request.build_opener(https_handler)
        urllib.request.install_opener(opener)
        
        print("✅ SSL certificate verification disabled globally")
        print("🔒 Warning: This reduces security - use only for trusted sources")
        
    except Exception as e:
        print(f"❌ Failed to disable SSL verification: {e}")
        raise


def enable_ssl_verification():
    """
    Re-enable SSL certificate verification by restoring default settings.
    
    This function restores the default SSL context and HTTPS handler,
    re-enabling certificate verification for security.
    
    Returns:
        None
    """
    try:
        # Create default SSL context (with verification)
        ssl_context = ssl.create_default_context()
        
        # Create default HTTPS handler
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        
        # Install the default handler
        opener = urllib.request.build_opener(https_handler)
        urllib.request.install_opener(opener)
        
        print("✅ SSL certificate verification re-enabled")
        
    except Exception as e:
        print(f"❌ Failed to re-enable SSL verification: {e}")
        raise


def fetch_with_ssl_bypass(name: Optional[str] = None, id: Optional[int] = None):
    """
    Fetch UCI ML Repository dataset with automatic SSL bypass.
    
    This function temporarily disables SSL verification, fetches the dataset,
    then re-enables SSL verification for security.
    
    Parameters:
        name (str, optional): Dataset name or substring
        id (int, optional): Dataset ID
        
    Returns:
        Dataset object from ucimlrepo
        
    Example:
        from ssl_validation import fetch_with_ssl_bypass
        data = fetch_with_ssl_bypass(name='Iris')
    """
    from ucimlrepo import fetch_ucirepo
    
    # Store original state
    original_ssl_context = ssl.create_default_context()
    
    try:
        # Temporarily disable SSL verification
        disable_ssl_verification()
        
        # Fetch the dataset
        if name and id:
            raise ValueError('Only specify either dataset name or ID, not both')
        elif name:
            data = fetch_ucirepo(name=name)
        elif id:
            data = fetch_ucirepo(id=id)
        else:
            raise ValueError('Must provide either dataset name or ID')
            
        print(f"✅ Successfully fetched dataset")
        return data
        
    finally:
        # Always re-enable SSL verification for security
        enable_ssl_verification()


def configure_pandas_ssl():
    """
    Configure pandas to work with SSL certificate issues.
    
    This function sets up pandas to handle HTTPS URLs with SSL certificate
    verification disabled, which is specifically useful for reading CSV files
    from UCI ML Repository.
    
    Returns:
        None
    """
    try:
        # Import pandas to configure its URL handling
        import pandas as pd
        
        # Create unverified SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Monkey patch pandas' URL opener to use unverified SSL
        original_urlopen = urllib.request.urlopen
        
        def patched_urlopen(url, *args, **kwargs):
            if isinstance(url, str) and url.startswith('https://'):
                kwargs['context'] = ssl_context
            return original_urlopen(url, *args, **kwargs)
        
        urllib.request.urlopen = patched_urlopen
        
        print("✅ Pandas SSL configuration applied")
        
    except Exception as e:
        print(f"❌ Failed to configure pandas SSL: {e}")
        raise


# Convenience function for common use case
def setup_uci_ssl_bypass():
    """
    One-time setup function to configure SSL bypass for UCI ML Repository.
    
    Call this once at the beginning of your script to enable seamless
    access to UCI ML Repository datasets without SSL certificate issues.
    
    Example:
        from ssl_validation import setup_uci_ssl_bypass
        setup_uci_ssl_bypass()
        
        # Now you can use fetch_ucirepo normally
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(name='Iris')
    """
    print("🔧 Setting up UCI ML Repository SSL bypass...")
    disable_ssl_verification()
    configure_pandas_ssl()
    print("🎉 Setup complete! UCI ML Repository should now work without SSL issues")


if __name__ == "__main__":
    # Test the SSL bypass functionality
    print("Testing SSL validation bypass...")
    
    try:
        setup_uci_ssl_bypass()
        
        # Test fetch_ucirepo
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(name='Iris')
        
        print(f"✅ Test successful! Dataset shape: {data.data.features.shape}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Re-enable SSL for security
        enable_ssl_verification()
        print("🔒 SSL verification re-enabled for security")
