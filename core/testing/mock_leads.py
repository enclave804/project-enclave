"""
Mock leads for pipeline testing.

These leads match the Enclave Guard vertical ICP and use RFC 2606 reserved
.example.com domains to prevent any accidental collisions with real data.
"""

MOCK_LEADS: list[dict] = [
    {
        "contact": {
            "name": "Marcus Chen",
            "email": "mchen@test-finpay.example.com",
            "title": "CTO",
            "apollo_id": "test_apollo_person_001",
            "seniority": "c_suite",
            "linkedin_url": "https://linkedin.com/in/test-marcus-chen",
        },
        "company": {
            "name": "FinPay Solutions",
            "domain": "test-finpay.example.com",
            "industry": "fintech",
            "employee_count": 45,
            "apollo_id": "test_apollo_org_001",
            "website_url": "https://test-finpay.example.com",
            "tech_stack": {
                "WordPress": "5.8",
                "PHP": "7.4",
                "MySQL": "5.7",
                "AWS": "cloud",
                "Cloudflare": "cdn",
            },
        },
    },
    {
        "contact": {
            "name": "Dr. Sarah Okafor",
            "email": "sokafor@test-healthbridge.example.com",
            "title": "CEO",
            "apollo_id": "test_apollo_person_002",
            "seniority": "founder",
            "linkedin_url": "https://linkedin.com/in/test-sarah-okafor",
        },
        "company": {
            "name": "HealthBridge AI",
            "domain": "test-healthbridge.example.com",
            "industry": "healthcare",
            "employee_count": 25,
            "apollo_id": "test_apollo_org_002",
            "website_url": "https://test-healthbridge.example.com",
            "tech_stack": {
                "React": "18",
                "Node.js": "16",
                "PostgreSQL": "14",
                "Heroku": "cloud",
                "Stripe": "payments",
            },
        },
    },
    {
        "contact": {
            "name": "James Whitfield",
            "email": "jwhitfield@test-lexguard.example.com",
            "title": "CISO",
            "apollo_id": "test_apollo_person_003",
            "seniority": "director",
            "linkedin_url": "https://linkedin.com/in/test-james-whitfield",
        },
        "company": {
            "name": "LexGuard Legal Tech",
            "domain": "test-lexguard.example.com",
            "industry": "legal",
            "employee_count": 200,
            "apollo_id": "test_apollo_org_003",
            "website_url": "https://test-lexguard.example.com",
            "tech_stack": {
                "Apache": "2.4",
                "Java": "11",
                "Oracle DB": "19c",
                "FTP": "vsftpd",
                "Windows Server": "2016",
            },
        },
    },
]
