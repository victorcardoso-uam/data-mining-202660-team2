import datetime

TEAM_REGISTRY = {
    "cohort": "Team 2",
    "repository": "data-mining-202660-team2",
    "members": [
        {
            "name": "Oliver Pacheco",
            "student_id": "00458380",
            "role": "Lead Data Engineer",
            "assigned_reviewer": "Valeria Hurtado",
            "git_feature_branch": "feature/activity-10-oliver",
            "preferred_ai_assistant": "ChatGPT (GPT-4o)",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        {
            "name": "Valeria Hurtado",
            "student_id": "00487453",
            "role": "ML Engineer",
            "assigned_reviewer": "Brenda Elisa Cabrera Cruz",
            "git_feature_branch": "feature/activity-10-valeria-hurtado",
            "preferred_ai_assistant": "Claude",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    ]
}

def display_team_roster():
    print(f"\n{'='*20} {TEAM_REGISTRY['cohort']} ACTIVE ROSTER {'='*20}")
    for m in TEAM_REGISTRY["members"]:
        print(f"* {m['name']} ({m['student_id']}) | Role: {m['role']} | Branch: {m['git_feature_branch']}")
    print('='*60 + '\n')

if __name__ == '__main__':
    display_team_roster()
    