package authctx

import (
	"context"
)

type contextKey string

const principalContextKey contextKey = "auth-principal"

// Principal identifies the authenticated caller.
type Principal struct {
	ID        string
	Name      string
	Managed   bool
	BudgetUSD *float64
}

// WithPrincipal stores the authenticated caller in context.
func WithPrincipal(ctx context.Context, principal Principal) context.Context {
	return context.WithValue(ctx, principalContextKey, principal)
}

// PrincipalFromContext returns the authenticated caller from context.
func PrincipalFromContext(ctx context.Context) (Principal, bool) {
	principal, ok := ctx.Value(principalContextKey).(Principal)
	return principal, ok
}
