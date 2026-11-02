# Route a support message

**Task:** `T-ROUTE`

## Task and when to use

Send each customer message to the one of 77 banking intents that fits it.

## Recommended model

- **Recommended:** `gemini-4-argon` — $1.57 per 1,000 items.
- **Fallback:** none yet; only one model is good enough at this sample size.

The rule that chose them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Prompt (version `v1`)

```
You route a customer-service message for a retail bank to exactly one Banking77 intent.
Choose one label from this fixed list, copied exactly:
- activate_my_card
- age_limit
- apple_pay_or_google_pay
- atm_support
- automatic_top_up
- balance_not_updated_after_bank_transfer
- balance_not_updated_after_cheque_or_cash_deposit
- beneficiary_not_allowed
- cancel_transfer
- card_about_to_expire
- card_acceptance
- card_arrival
- card_delivery_estimate
- card_linking
- card_not_working
- card_payment_fee_charged
- card_payment_not_recognised
- card_payment_wrong_exchange_rate
- card_swallowed
- cash_withdrawal_charge
- cash_withdrawal_not_recognised
- change_pin
- compromised_card
- contactless_not_working
- country_support
- declined_card_payment
- declined_cash_withdrawal
- declined_transfer
- direct_debit_payment_not_recognised
- disposable_card_limits
- edit_personal_details
- exchange_charge
- exchange_rate
- exchange_via_app
- extra_charge_on_statement
- failed_transfer
- fiat_currency_support
- get_disposable_virtual_card
- get_physical_card
- getting_spare_card
- getting_virtual_card
- lost_or_stolen_card
- lost_or_stolen_phone
- order_physical_card
- passcode_forgotten
- pending_card_payment
- pending_cash_withdrawal
- pending_top_up
- pending_transfer
- pin_blocked
- receiving_money
- Refund_not_showing_up
- request_refund
- reverted_card_payment?
- supported_cards_and_currencies
- terminate_account
- top_up_by_bank_transfer_charge
- top_up_by_card_charge
- top_up_by_cash_or_cheque
- top_up_failed
- top_up_limits
- top_up_reverted
- topping_up_by_card
- transaction_charged_twice
- transfer_fee_charged
- transfer_into_account
- transfer_not_received_by_recipient
- transfer_timing
- unable_to_verify_identity
- verify_my_identity
- verify_source_of_funds
- verify_top_up
- virtual_card_not_working
- visa_or_mastercard
- why_verify_identity
- wrong_amount_of_cash_received
- wrong_exchange_rate_for_cash_withdrawal

Reply with one JSON object and nothing else: {"intent": "<one label from the list>"}.
Never invent a label. Never return more than one label.

Customer message:
{text}
```

## Output schema

```json
{
  "additionalProperties": false,
  "description": "T-ROUTE: exactly one Banking77 intent label.",
  "properties": {
    "intent": {
      "minLength": 1,
      "title": "Intent",
      "type": "string"
    }
  },
  "required": [
    "intent"
  ],
  "title": "RouteOut",
  "type": "object"
}
```

## Guardrails

- An `unclear` or abstained answer goes to a human; it is scored as wrong, so abstaining is never free.
- The schema-valid rate must stay at or above 98%.
- Never auto-close, auto-send or auto-approve a model answer; a person makes the final call.

## Re-check

`wam recheck run --tasks T-ROUTE --against results/runs/live` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).
