# Brightleaf Solar IT Security Policy (fictional sample)

This is a fictional policy used to demonstrate DocuMind.

## Passwords

Passwords must be at least 14 characters long. Employees should use passphrases made of several unrelated words. Passwords must never be reused across systems, and all employees must store passwords in the company-provided 1Password vault. Passwords do not expire on a schedule, but they must be changed immediately if a compromise is suspected.

## Multi-Factor Authentication

Multi-factor authentication (MFA) is mandatory for email, the VPN, the HR portal, and all cloud consoles. Hardware security keys such as YubiKeys are required for engineers with production access. SMS codes are not an approved MFA method because they can be intercepted through SIM swapping.

## Laptops and Devices

Company laptops are encrypted with full-disk encryption and managed through the device management system. Screens must lock automatically after 5 minutes of inactivity. Operating system updates must be installed within 7 days of release.

Personal devices may be used to access email and chat only if they are enrolled in device management. Company data must never be stored on personal USB drives.

Lost or stolen devices must be reported to the IT help desk within 1 hour so the device can be remotely wiped.

## Phishing and Suspicious Email

If you receive a suspicious email, do not click links or open attachments. Use the "Report Phishing" button in your email client, which sends the message to the security team. The security team runs simulated phishing tests every quarter.

## Security Incidents

A security incident is any event that could compromise the confidentiality, integrity, or availability of company data. Examples include malware infections, unauthorized access, and accidental data sharing.

Report suspected incidents immediately to security@brightleaf.example or through the #security-incidents Slack channel. The security on-call engineer responds within 30 minutes, 24 hours a day. Do not try to investigate or fix the incident yourself, and do not delete any evidence.

## Data Classification

Brightleaf classifies data into four levels: Public, Internal, Confidential, and Restricted. Customer personal data and payment information are Restricted. Restricted data may only be stored in approved systems and must never be pasted into public AI tools or chatbots.

## Access Reviews

Managers review their team's access to systems every quarter. Access for employees who leave the company is removed on their last working day.
