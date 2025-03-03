#!/usr/bin/env python3
"""
Generate Sample Data - Creates sample CSV files for the multi-agent system
"""

import os
import csv
import random
import argparse
from datetime import datetime, timedelta

def ensure_dir(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_billing_data(output_dir):
    """Generate sample billing data."""
    print("Generating billing data...")
    
    # Billing data
    billing_file = os.path.join(output_dir, "billing_data.csv")
    with open(billing_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "content", "category", "date"])
        
        billing_contents = [
            "The Basic Plan costs $49 per month and includes 5GB data, 100 minutes, and unlimited texts.",
            "The Standard Plan costs $79 per month and includes 15GB data, 500 minutes, and unlimited texts.",
            "The Premium Plan costs $129 per month and includes 50GB data, unlimited minutes, and unlimited texts.",
            "The Family Plan costs $199 per month and includes 100GB shared data, unlimited minutes, and unlimited texts for up to 5 lines.",
            "The Business Plan costs $299 per month and includes 200GB shared data, unlimited minutes, and unlimited texts for up to 10 lines.",
            "The Enterprise Plan costs $599 per month and includes 500GB shared data, unlimited minutes, and unlimited texts for up to 25 lines.",
            "The Ultimate Plan costs $999 per month and includes unlimited data, unlimited minutes, and unlimited texts for up to 50 lines.",
            "The $331 Premium Plus Plan includes unlimited data, unlimited minutes, unlimited texts, free international roaming, and priority customer support.",
            "All plans are billed monthly and include taxes and fees.",
            "Late payments incur a $15 fee and may result in service interruption.",
            "Customers can upgrade their plan at any time without penalty.",
            "Downgrading a plan takes effect at the end of the current billing cycle.",
            "Plan prices are subject to change with 30 days notice.",
            "Refunds are processed within 7-10 business days.",
            "Credit card is the preferred payment method for all plans."
        ]
        
        for i, content in enumerate(billing_contents):
            category = random.choice(["plans", "payments", "refunds", "general"])
            date = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
            writer.writerow([i+1, content, category, date])
    
    # Payment FAQ
    payment_faq_file = os.path.join(output_dir, "payment_faq.csv")
    with open(payment_faq_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question", "answer"])
        
        faqs = [
            ("How do I update my billing information?", "You can update your billing information by logging into your account, going to Settings > Billing, and selecting 'Update Payment Method'."),
            ("When will I be charged?", "Charges are processed on the same day each month based on your signup date."),
            ("Can I change my billing date?", "Yes, you can change your billing date once every 3 months by contacting customer support."),
            ("Do you offer refunds?", "We offer full refunds within 14 days of purchase. After that, refunds are prorated based on usage."),
            ("How do I cancel my subscription?", "To cancel your subscription, go to Settings > Billing > Cancel Subscription. Follow the prompts to confirm."),
            ("Can I get an invoice for my business?", "Yes, you can generate invoices for your business by enabling this feature in your account settings."),
            ("How do I check my current plan?", "Your current plan is displayed at the top of your dashboard or in Settings > Billing > Current Plan."),
            ("What happens if my payment fails?", "If your payment fails, we'll retry 3 times over 7 days before suspending service. You'll receive email notifications."),
            ("Do you offer discounts?", "We offer annual payment discounts of 15% and referral discounts of $10 per successful referral."),
            ("What payment methods do you accept?", "We accept all major credit cards, PayPal, and bank transfers for annual plans.")
        ]
        
        for i, (question, answer) in enumerate(faqs):
            writer.writerow([i+1, question, answer])
    
    print(f"Created {billing_file} and {payment_faq_file}")

def generate_technical_data(output_dir):
    """Generate sample technical data."""
    print("Generating technical data...")
    
    # Technical data
    technical_file = os.path.join(output_dir, "technical_data.csv")
    with open(technical_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "content", "category", "date"])
        
        technical_contents = [
            "To reset your password, click on 'Forgot Password' on the login screen and follow the instructions sent to your email.",
            "The app requires at least iOS 14 or Android 8.0 to function correctly.",
            "Location services must be enabled for the map feature to work properly.",
            "Data sync issues can usually be resolved by logging out and logging back in.",
            "To backup your data, go to Settings > Backup & Sync and toggle 'Auto Backup' to ON.",
            "The desktop application is compatible with Windows 10/11 and macOS 10.15 or later.",
            "To enable dark mode, go to Settings > Display > Theme and select 'Dark'.",
            "If the app is crashing, try clearing the cache in Settings > Storage > Clear Cache.",
            "Two-factor authentication can be enabled in Settings > Security > 2FA.",
            "Bluetooth connectivity issues can often be resolved by turning Bluetooth off and on again.",
            "To update the app, visit the App Store or Google Play Store and check for updates.",
            "The system automatically logs you out after 30 minutes of inactivity for security reasons.",
            "Video calls require at least 2 Mbps upload and download speed for HD quality.",
            "To export your data, go to Settings > Privacy > Export Data and select the desired format.",
            "The maximum file upload size is 50MB per file."
        ]
        
        for i, content in enumerate(technical_contents):
            category = random.choice(["account", "app", "device", "connectivity", "general"])
            date = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
            writer.writerow([i+1, content, category, date])
    
    # Product documentation
    product_docs_file = os.path.join(output_dir, "product_docs.csv")
    with open(product_docs_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "title", "section", "documentation"])
        
        docs = [
            ("Getting Started", "Installation", "Download the app from the App Store or Google Play Store. Open the app and follow the on-screen instructions to create an account."),
            ("Getting Started", "Account Setup", "After installation, you'll need to create an account with your email and a strong password. Verify your email to activate all features."),
            ("Features", "Dashboard", "The dashboard provides an overview of your activity, recent notifications, and quick access to frequently used features."),
            ("Features", "Messaging", "Send text, images, and files to other users. Group messaging supports up to 50 participants."),
            ("Features", "Calling", "Make voice and video calls to other users. Conference calls support up to 8 participants simultaneously."),
            ("Security", "Encryption", "All messages and calls are end-to-end encrypted. Your data is only accessible to you and your intended recipients."),
            ("Security", "Two-Factor Authentication", "Enable 2FA for additional account security. We support SMS verification and authenticator apps."),
            ("Troubleshooting", "Connection Issues", "If you're experiencing connection problems, check your internet connection, restart the app, or try switching between Wi-Fi and cellular data."),
            ("Troubleshooting", "Audio/Video Problems", "If you're having audio or video issues during calls, check your device permissions and make sure your microphone and camera are not being used by other apps."),
            ("Advanced", "API Integration", "Developers can integrate with our API for custom implementations. Visit our developer portal for documentation and authentication keys.")
        ]
        
        for i, (title, section, documentation) in enumerate(docs):
            writer.writerow([i+1, title, section, documentation])
    
    # Technical FAQ
    technical_faq_file = os.path.join(output_dir, "technical_faq.csv")
    with open(technical_faq_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question", "answer"])
        
        faqs = [
            ("How do I reset my password?", "To reset your password, click 'Forgot Password' on the login screen, enter your email, and follow the instructions sent to your inbox."),
            ("Why is the app crashing?", "App crashes can be caused by outdated versions, insufficient storage, or conflicting apps. Try updating the app, clearing cache, or restarting your device."),
            ("How do I enable notifications?", "Go to Settings > Notifications and toggle 'Allow Notifications' to ON. You can customize notification types in this same menu."),
            ("Can I use the app offline?", "Some features work offline, but core functionality requires an internet connection. Offline mode allows viewing previously loaded content only."),
            ("How do I transfer data between devices?", "Sign in with the same account on both devices. Go to Settings > Backup & Sync on the old device, create a backup, then restore it on the new device."),
            ("Is my data secure?", "Yes, we use end-to-end encryption for all data. Your information is encrypted on your device and can only be decrypted by the intended recipient."),
            ("How much storage does the app use?", "The app typically uses 150-300MB of storage initially. This can grow to 1-2GB depending on usage and cached data. You can clear cache in Settings."),
            ("Does the app work on tablets?", "Yes, the app is optimized for both phones and tablets, with adaptive layouts that take advantage of larger screens."),
            ("How do I link multiple accounts?", "Go to Settings > Accounts > Link Account and follow the prompts to connect additional accounts to your primary profile."),
            ("What permissions does the app need?", "The app requires camera, microphone, and storage permissions for core functionality. Location, contacts, and calendar access are optional features.")
        ]
        
        for i, (question, answer) in enumerate(faqs):
            writer.writerow([i+1, question, answer])
    
    print(f"Created {technical_file}, {product_docs_file}, and {technical_faq_file}")

def generate_order_data(output_dir):
    """Generate sample order data."""
    print("Generating order data...")
    
    # Order data
    order_file = os.path.join(output_dir, "order_data.csv")
    with open(order_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "content", "category", "date"])
        
        order_contents = [
            "Order #12345 was placed on 2025-01-15 and includes 3 items totaling $79.99.",
            "Order #23456 was placed on 2025-01-20 and includes 1 item totaling $129.99.",
            "Order #34567 was placed on 2025-01-25 and includes 5 items totaling $299.95.",
            "Order #45678 was placed on 2025-01-30 and includes 2 items totaling $59.98.",
            "Order #56789 was placed on 2025-02-05 and includes 4 items totaling $149.96.",
            "Standard shipping takes 3-5 business days and costs $5.99.",
            "Express shipping takes 1-2 business days and costs $15.99.",
            "Overnight shipping is guaranteed next business day and costs $29.99.",
            "International shipping takes 7-14 business days and costs $24.99.",
            "All orders are processed within 24 hours of being placed.",
            "Orders placed after 2 PM may not be processed until the next business day.",
            "Order tracking information is sent via email once the order has shipped.",
            "Shipping to PO boxes is available for standard shipping only.",
            "Orders over $100 qualify for free standard shipping.",
            "Backorders typically ship within 3-4 weeks of the order date."
        ]
        
        for i, content in enumerate(order_contents):
            category = random.choice(["orders", "shipping", "tracking", "general"])
            date = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
            writer.writerow([i+1, content, category, date])
    
    # Shipping policies
    shipping_file = os.path.join(output_dir, "shipping_policies.csv")
    with open(shipping_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["policy_id", "policy_title", "policy_text"])
        
        policies = [
            ("Domestic Shipping", "Domestic orders (within the continental US) are shipped via our preferred carriers and typically take 3-5 business days to arrive. Orders over $100.00 qualify for free standard shipping. Express (1-2 days) and overnight shipping options are available at checkout for an additional fee."),
            ("International Shipping", "International shipping is available to most countries. Delivery times vary by location but typically range from 7-14 business days. Import duties, taxes, and customs fees are the responsibility of the recipient and are not included in the shipping cost."),
            ("Shipping Restrictions", "We cannot ship to P.O. boxes for express or overnight deliveries. Some products cannot be shipped internationally due to regulations. Hazardous materials are subject to special shipping requirements and may incur additional fees."),
            ("Order Processing", "All orders are processed within 1-2 business days. Orders placed after 2:00 PM EST may not be processed until the following business day. During peak periods (holidays), processing may take up to 3 business days."),
            ("Tracking Information", "Tracking information is automatically sent to the email address provided at checkout once your order has shipped. You can also view tracking information by logging into your account and viewing your order history."),
            ("Shipping Delays", "Weather, natural disasters, customs clearance, and carrier delays can affect delivery times. We are not responsible for shipping delays beyond our control. During holiday seasons (November-December), please allow extra time for delivery."),
            ("Address Accuracy", "Customers are responsible for providing accurate shipping information. We are not responsible for orders shipped to incorrect addresses provided by customers. Address changes cannot be made once an order has shipped."),
            ("Lost or Damaged Packages", "If your package is lost or damaged during transit, please contact customer service within 5 business days of the expected delivery date. We will work with the carrier to resolve the issue or process a replacement shipment."),
            ("Shipping Insurance", "All orders are automatically insured against loss or damage during transit up to $100. Additional insurance can be purchased at checkout for orders exceeding this value."),
            ("Multiple Shipments", "Orders containing multiple items may ship in separate packages, depending on inventory location and availability. You will receive tracking information for each package shipped.")
        ]
        
        for i, (title, text) in enumerate(policies):
            writer.writerow([i+1, title, text])
    
    # Return policies
    return_file = os.path.join(output_dir, "return_policies.csv")
    with open(return_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["policy_id", "policy_title", "policy_text"])
        
        policies = [
            ("Return Eligibility", "Items must be returned within 30 days of delivery in their original condition with all packaging and tags intact. Used, damaged, or altered items are not eligible for return unless they arrived damaged or defective."),
            ("Return Process", "To initiate a return, log into your account, go to Order History, select the order containing the item you wish to return, and click 'Return Item'. Follow the on-screen instructions to generate a return shipping label."),
            ("Return Shipping", "For standard returns, customers are responsible for return shipping costs. For defective items or shipping errors, we will provide a prepaid return shipping label. Return shipping costs are non-refundable."),
            ("Refund Processing", "Refunds are processed within 5-7 business days after the returned item is received and inspected. Refunds are issued to the original payment method used for the purchase. Shipping charges are not refunded except for defective items."),
            ("Exchanges", "We offer direct exchanges for the same item in a different size or color, subject to availability. To request an exchange, follow the same process as returns but select 'Exchange' instead of 'Return' during the return process."),
            ("Gift Returns", "Items received as gifts can be returned for store credit or exchanged for another item. The return must be accompanied by the order number or gift receipt. Store credit will be issued to the gift recipient, not the original purchaser."),
            ("Non-Returnable Items", "Certain items cannot be returned for health, safety, or regulatory reasons, including personal care items, undergarments, food products, customized or personalized items, and digital downloads. These items are clearly marked as non-returnable at checkout."),
            ("Damaged or Incorrect Items", "If you receive a damaged or incorrect item, please contact customer service within 48 hours of delivery. Please include photos of the damaged item or packaging. We will arrange for a replacement or refund."),
            ("Late Returns", "Returns initiated after the 30-day return period may be accepted at our discretion but will only be eligible for store credit. Returns initiated after 60 days will not be accepted under any circumstances."),
            ("Return Limitations", "We reserve the right to limit returns from customers with a pattern of excessive returns. Multiple returns of the same item or frequent returns may be subject to review and may result in limiting return privileges.")
        ]
        
        for i, (title, text) in enumerate(policies):
            writer.writerow([i+1, title, text])
    
    print(f"Created {order_file}, {shipping_file}, and {return_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate sample data for the multi-agent system")
    parser.add_argument("--output-dir", type=str, default="data/raw_data", help="Output directory for the CSV files")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    ensure_dir(output_dir)
    
    generate_billing_data(output_dir)
    generate_technical_data(output_dir)
    generate_order_data(output_dir)
    
    print(f"All sample data files created in {output_dir}")
    print("Now you can run the data ingestion script to create FAISS indices.")

if __name__ == "__main__":
    main()